#include "trellis_sched.h"

#include "trellis_debug.h"
#include "trellis_model.h"

#include "ggml-cpu.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

namespace trellis {

namespace {

int  g_sched_mode  = -1;   // -1 auto, 0 off, 1 on
int  g_vulkan_fallback_mode = -1; // -1 environment, 0 off, 1 on

// Match the maximum DiT and sparse graph capacity.
constexpr size_t kSchedGraphSize = 65536;

std::once_flag  g_cpu_once;
ggml_backend_t  g_cpu = nullptr;

std::once_flag              g_vulkan_once;
std::vector<ggml_backend_t> g_vulkan; // Vulkan fallbacks for an HTP primary

int resolve_mode() {
    if (g_sched_mode >= 0) return g_sched_mode;
    if (const char* e = std::getenv("TRELLIS_SCHED")) return atoi(e) > 0 ? 1 : 0;
    return -1;   // still auto
}

bool vulkan_fallback_enabled() {
    if (g_vulkan_fallback_mode >= 0) return g_vulkan_fallback_mode > 0;
    const char* value = std::getenv("TRELLIS_VULKAN_FALLBACK");
    return value && std::atoi(value) > 0;
}

}  // namespace

void set_sched_mode(int mode) { g_sched_mode = mode; }
void set_vulkan_fallback_mode(int mode) { g_vulkan_fallback_mode = mode; }

ggml_backend_t fallback_cpu_backend() {
    std::call_once(g_cpu_once, [] {
        g_cpu = ggml_backend_cpu_init();
        if (g_cpu) tune_cpu_threads(g_cpu);
    });
    return g_cpu;
}

const std::vector<ggml_backend_t>& vulkan_backends() {
    std::call_once(g_vulkan_once, [] {
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t d = ggml_backend_dev_get(i);
            ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(d);
            const char* reg_name = reg ? ggml_backend_reg_name(reg) : "";
            if (std::strcmp(reg_name, "Vulkan") != 0) continue;
            if (ggml_backend_t b = ggml_backend_dev_init(d, nullptr)) {
                fprintf(stderr, "[trellis] Vulkan fallback %s (%s)\n",
                        ggml_backend_name(b), ggml_backend_dev_description(d));
                g_vulkan.push_back(b);
            }
        }
    });
    return g_vulkan;
}

bool sched_enabled_for(ggml_backend_t primary) {
    const int mode = resolve_mode();
    if (mode == 0) return false;
    if (mode == 1) return true;
    // Auto-enable only for the partial-coverage HTP backend. CUDA and Vulkan retain
    // their existing direct execution path unless the user explicitly requests a scheduler.
    return primary && std::strncmp(ggml_backend_name(primary), "HTP", 3) == 0;
}

ggml_backend_sched_t make_sched(ggml_backend_t primary) {
    if (!primary) return nullptr;

    std::vector<ggml_backend_t> backends;

    if (!sched_enabled_for(primary)) return nullptr;
    backends.push_back(primary);   // owns the weights and has highest priority
    if (vulkan_fallback_enabled() &&
        std::strncmp(ggml_backend_name(primary), "HTP", 3) == 0) {
        for (ggml_backend_t b : vulkan_backends()) backends.push_back(b);
    }
    if (!ggml_backend_is_cpu(primary)) {
        if (ggml_backend_t cpu = fallback_cpu_backend()) backends.push_back(cpu);
    }

    if (backends.size() < 2) {
        fprintf(stderr, "[trellis] only one backend available; skipping scheduler\n");
        return nullptr;
    }

    ggml_backend_sched_t s = ggml_backend_sched_new(
        backends.data(), /*bufts=*/nullptr, (int) backends.size(),
        kSchedGraphSize, /*parallel=*/false, /*op_offload=*/true);

    if (!s) {
        fprintf(stderr, "[trellis] failed to create backend scheduler; using %s alone\n", ggml_backend_name(primary));
        return nullptr;
    }

    fprintf(stderr, "[trellis] scheduler:");
    for (ggml_backend_t b : backends) fprintf(stderr, " %s", ggml_backend_name(b));
    fprintf(stderr, "  (ops fall back in that order)\n");
    return s;
}

GraphExec::GraphExec(const Model& m)
    : backend_(m.backend), sched_(m.sched) {}

GraphExec::~GraphExec() {
    if (galloc_) ggml_gallocr_free(galloc_);
    // sched_ is owned by the Model, not by us.
}

bool GraphExec::alloc(ggml_cgraph* graph) {
    if (sched_) {
        // Clears the previous graph's assignment; without it the scheduler keeps the old
        // allocation and the freshly uploaded inputs would land in stale buffers.
        ggml_backend_sched_reset(sched_);
        if (!ggml_backend_sched_alloc_graph(sched_, graph)) {
            fprintf(stderr, "[trellis] scheduler failed to allocate graph\n");
            return false;
        }
        return true;
    }
    if (!galloc_) galloc_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
    return ggml_gallocr_alloc_graph(galloc_, graph);
}

ggml_status GraphExec::compute(ggml_cgraph* graph, const char* label) {
    if (sched_) return compute_graph_sched(sched_, graph, label);
    return compute_graph(backend_, graph, label);
}

size_t GraphExec::buffer_size() const {
    if (galloc_) return ggml_gallocr_get_buffer_size(galloc_, 0);
    return 0;
}

}  // namespace trellis
