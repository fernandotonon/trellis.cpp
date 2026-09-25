#include "trellis_model.h"
#include "trellis_debug.h"
#include "trellis_sched.h"

#include "ggml.h"
#include "gguf.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#ifdef TRELLIS_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <cctype>
#include <cstdint>
#include <thread>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace {
// plain fseek()'s offset is a 32-bit `long` under MSVC even in 64-bit builds,
// so it silently truncates offsets past 2GB -- fatal for the flow GGUFs here,
// which run ~2.4GB.
int trellis_fseek64(FILE* f, int64_t offset, int origin) {
#ifdef _WIN32
    return _fseeki64(f, offset, origin);
#else
    return fseeko(f, (off_t) offset, origin);
#endif
}
}  // namespace

namespace trellis {

bool g_require_gpu = false;   // --require-gpu; set by trellis_run
int  g_cpu_threads = 0;       // --threads; 0 = all cores. Set by trellis_run.
std::string g_backend;        // --backend NAME;  set by trellis_run

// Tune both an explicitly selected CPU backend and the scheduler's CPU fallback.
void tune_cpu_threads(ggml_backend* b) {
    if (!b || !ggml_backend_is_cpu(b)) return;
    int n = g_cpu_threads;
    if (n <= 0) {
        if (const char* e = std::getenv("TRELLIS_THREADS")) n = atoi(e);
    }
    if (n <= 0) n = (int) std::thread::hardware_concurrency();
    if (n <= 0) n = 4;
    ggml_backend_cpu_set_n_threads(b, n);
    fprintf(stderr, "[trellis] CPU backend: %d threads\n", n);
}

// --backend, else TRELLIS_BACKEND (so the test binaries, which don't parse args,
// can select a backend too).
static std::string requested_backend() {
    if (!g_backend.empty()) return g_backend;
    if (const char* e = std::getenv("TRELLIS_BACKEND")) return e;
    return {};
}

static bool name_matches(const char* have, const std::string& want) {
    if (!have) return false;
    if (std::strlen(have) != want.size()) return false;
    for (size_t i = 0; i < want.size(); ++i) {
        if (std::tolower((unsigned char) have[i]) != std::tolower((unsigned char) want[i]))
            return false;
    }
    return true;
}

// Devices whose ggml backend registration (e.g. "Vulkan", "Hexagon", "CPU") or device
// name (e.g. "Vulkan0") matches `want`, in registration order.
static std::vector<ggml_backend_dev_t> devices_named(const std::string& want) {
    std::vector<ggml_backend_dev_t> out;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (!d) continue;
        if (name_matches(ggml_backend_reg_name(ggml_backend_dev_backend_reg(d)), want) ||
            name_matches(ggml_backend_dev_name(d), want)) {
            out.push_back(d);
        }
    }
    return out;
}

static void list_backends(FILE* f) {
    fprintf(f, "[trellis] available backends:");
    for (size_t i = 0; i < ggml_backend_reg_count(); ++i)
        fprintf(f, " %s", ggml_backend_reg_name(ggml_backend_reg_get(i)));
    fprintf(f, "\n");
}

static ggml_backend* make_backend(int gpu) {
    // An explicit --backend wins over everything below. This matters on Snapdragon,
    // where the Hexagon NPU and the Adreno GPU both register as
    // GGML_BACKEND_DEVICE_TYPE_GPU: the "largest reported memory" heuristic further
    // down cannot tell them apart in any meaningful way, so the choice has to be
    // stated rather than guessed.
    if (const std::string want = requested_backend(); !want.empty()) {
        std::vector<ggml_backend_dev_t> devs = devices_named(want);
        if (devs.empty()) {
            fprintf(stderr, "[trellis] no backend named '%s' in this build\n", want.c_str());
            list_backends(stderr);
            throw std::runtime_error("[trellis] unknown backend: " + want);
        }
        const size_t idx = (gpu > 0 && (size_t) gpu < devs.size()) ? (size_t) gpu : 0;
        if (gpu > 0 && idx != (size_t) gpu)
            fprintf(stderr, "[trellis] --gpu %d out of range for backend '%s' (%zu device(s)); using 0\n",
                    gpu, want.c_str(), devs.size());
        ggml_backend_dev_props pr; ggml_backend_dev_get_props(devs[idx], &pr);
        if (ggml_backend* b = ggml_backend_dev_init(devs[idx], nullptr)) {
            fprintf(stderr, "[trellis] using %s -- %s (%zu MB)\n",
                    ggml_backend_name(b), pr.description ? pr.description : "?",
                    pr.memory_total / (1024 * 1024));
            tune_cpu_threads(b);          // --backend CPU lands here too
            return b;
        }
        throw std::runtime_error("[trellis] backend '" + want + "' failed to initialize");
    }

    // gpu < 0 is an explicit request for CPU.
    if (gpu < 0) { ggml_backend* b = ggml_backend_cpu_init(); tune_cpu_threads(b); return b; }
#ifdef TRELLIS_USE_CUDA
    {
        ggml_backend* b = ggml_backend_cuda_init(gpu);
        if (b) return b;
        fprintf(stderr, "[trellis] CUDA init failed on device %d\n", gpu);
    }
#endif
    // Generic GPU path (e.g. Vulkan when built without CUDA). Enumerate GPU/IGPU
    // devices in backend order; `--gpu N` selects the N-th (matching the CUDA
    // path's index semantics — fixes #16, where --gpu was ignored on Vulkan). The
    // default `--gpu 0` keeps the "largest VRAM" heuristic: enumeration order can
    // put a small iGPU first, and the cascade is VRAM-hungry, so device 0 alone is
    // a poor default. An explicit index >0 is honored verbatim.
    {
        std::vector<ggml_backend_dev_t> gpus;
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t d = ggml_backend_dev_get(i);
            if (!d) continue;
            enum ggml_backend_dev_type t = ggml_backend_dev_type(d);
            // IGPU: integrated GPUs (e.g. Vulkan on a UMA APU) report a distinct type.
            if (t == GGML_BACKEND_DEVICE_TYPE_GPU || t == GGML_BACKEND_DEVICE_TYPE_IGPU)
                gpus.push_back(d);
        }
        ggml_backend_dev_t chosen = nullptr;
        size_t chosen_mem = 0;
        if (gpu > 0 && (size_t) gpu < gpus.size()) {
            chosen = gpus[(size_t) gpu];
            ggml_backend_dev_props pr; ggml_backend_dev_get_props(chosen, &pr);
            chosen_mem = pr.memory_total;
        } else if (!gpus.empty()) {
            if (gpu > 0)
                fprintf(stderr, "[trellis] --gpu %d out of range (%zu GPU device(s) found); using the largest\n",
                        gpu, gpus.size());
            for (ggml_backend_dev_t d : gpus) {
                ggml_backend_dev_props pr; ggml_backend_dev_get_props(d, &pr);
                if (pr.memory_total > chosen_mem) { chosen_mem = pr.memory_total; chosen = d; }
            }
        }
        if (chosen) {
            ggml_backend* b = ggml_backend_dev_init(chosen, nullptr);
            if (b) {
                fprintf(stderr, "[trellis] using %s (%zu MB)\n", ggml_backend_name(b), chosen_mem / (1024 * 1024));
                return b;
            }
        }
    }
    // A GPU was requested but none is usable. By default fall back to CPU
    // (preserves the original behavior). Opt in to strict GPU-only with
    // --require-gpu — then we throw rather than silently running the
    // VRAM-hungry cascade on the host (which balloons RAM and can OOM the box).
    if (g_require_gpu) {
        throw std::runtime_error(
            "[trellis] no usable GPU backend found and --require-gpu is set; refusing CPU fallback.");
    }
    fprintf(stderr, "[trellis] no GPU backend available; falling back to CPU\n");
    { ggml_backend* b = ggml_backend_cpu_init(); tune_cpu_threads(b); return b; }
}

// Reuse backend instances across sequential model stages. In particular, repeatedly
// releasing and reacquiring a FastRPC session can destabilize the Hexagon runtime.
static ggml_backend* shared_backend(int gpu) {
    static std::map<std::pair<int, std::string>, ggml_backend*> cache;
    const auto key = std::make_pair(gpu, requested_backend());
    if (auto it = cache.find(key); it != cache.end()) return it->second;
    ggml_backend* b = make_backend(gpu);
    cache[key] = b;
    return b;
}

// Conversions aimed at other runtimes store tensors in a quantization-friendly 2-D layout
// rather than their natural shape — a 1536-element RMS-norm gamma becomes [256, 6] instead of
// [128, 12], an [8, 1536] input projection becomes [256, 48] — and record the real shape in a
// metadata key ending `.orig_shape.<tensor name>` (ComfyUI-GGUF and its derivatives do this).
// The element order is untouched, so restoring the shape is a pure reinterpretation of the same
// bytes. Without it the layout looks plausible everywhere and only fails when a matmul finally
// compares widths, deep inside the graph.
//
// Only non-quantized types are restored: a quantized tensor's row length must stay a multiple
// of its block size, which reshaping would break.
static int restore_orig_shapes(gguf_context* gguf, ggml_context* meta) {
    static const char* MARK = ".orig_shape.";
    int fixed = 0;
    for (int64_t k = 0, nk = gguf_get_n_kv(gguf); k < nk; ++k) {
        const std::string key = gguf_get_key(gguf, k);
        const size_t p = key.find(MARK);
        if (p == std::string::npos) continue;
        if (gguf_get_kv_type(gguf, k) != GGUF_TYPE_ARRAY) continue;

        ggml_tensor* t = ggml_get_tensor(meta, key.substr(p + strlen(MARK)).c_str());
        if (!t || ggml_blck_size(t->type) != 1) continue;

        const size_t nd = gguf_get_arr_n(gguf, k);
        if (nd < 1 || nd > GGML_MAX_DIMS) continue;
        const void* raw = gguf_get_arr_data(gguf, k);
        const gguf_type at = gguf_get_arr_type(gguf, k);

        // The recorded shape is in torch order (outermost dimension first); ggml's ne is the
        // reverse, so read it back to front.
        int64_t ne[GGML_MAX_DIMS] = { 1, 1, 1, 1 };
        int64_t elems = 1;
        for (size_t d = 0; d < nd; ++d) {
            int64_t v;
            switch (at) {
                case GGUF_TYPE_INT32:  v = ((const int32_t*)  raw)[d]; break;
                case GGUF_TYPE_UINT32: v = ((const uint32_t*) raw)[d]; break;
                case GGUF_TYPE_INT64:  v = ((const int64_t*)  raw)[d]; break;
                case GGUF_TYPE_UINT64: v = (int64_t)((const uint64_t*) raw)[d]; break;
                default: v = -1;
            }
            if (v <= 0) { elems = -1; break; }
            ne[nd - 1 - d] = v;
            elems *= v;
        }
        if (elems != ggml_nelements(t)) continue;   // padded, not merely reshaped — leave it

        for (int d = 0; d < GGML_MAX_DIMS; ++d) t->ne[d] = ne[d];
        t->nb[0] = ggml_type_size(t->type);
        for (int d = 1; d < GGML_MAX_DIMS; ++d) t->nb[d] = t->nb[d - 1] * t->ne[d - 1];
        ++fixed;
    }
    return fixed;
}

Model Model::load(const std::string& path, int gpu) {
    ScopedTimer _t("load " + path.substr(path.find_last_of("/\\") + 1));
    Model m;

    ggml_context* meta = nullptr;
    gguf_init_params gp{};
    gp.no_alloc = true;       // tensors are metadata only; we upload data ourselves
    gp.ctx      = &meta;
    m.gguf = gguf_init_from_file(path.c_str(), gp);
    if (!m.gguf) throw std::runtime_error("failed to open gguf: " + path);
    m.meta = meta;

    // metadata
    if (int64_t k = gguf_find_key(m.gguf, "general.architecture"); k >= 0)
        m.arch = gguf_get_val_str(m.gguf, k);
    if (int64_t k = gguf_find_key(m.gguf, "trellis.config_json"); k >= 0)
        m.config_json = gguf_get_val_str(m.gguf, k);

    if (int fixed = restore_orig_shapes(m.gguf, meta))
        fprintf(stderr, "[trellis] %s: restored %d reshaped tensor(s) from orig_shape metadata\n",
                path.c_str(), fixed);

    m.backend = shared_backend(gpu);
    m.on_gpu  = !ggml_backend_is_cpu(m.backend);

    // Backends can expose an extra buffer type that performs device-specific weight
    // repacking during upload. Hexagon requires this for its tiled quantized matmuls.
    ggml_backend_buffer_type_t repack_buft = nullptr;
    {
        ggml_backend_dev_t dev = ggml_backend_get_device(m.backend);
        ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
        // CPU repacking requires a per-tensor support check that this loader does not yet
        // implement, so keep the generic CPU path unchanged.
        if (dev && ggml_backend_is_cpu(m.backend)) {
            if (verbose())
                fprintf(stderr, "[trellis:dbg] extra-bufts: skipped on the CPU backend\n");
            reg = nullptr;
        }
        void*              raw = reg ? ggml_backend_reg_get_proc_address(
                                           reg, "ggml_backend_dev_get_extra_bufts") : nullptr;
        if (raw) {
            auto fn = (ggml_backend_dev_get_extra_bufts_t) raw;
            if (ggml_backend_buffer_type_t* bufts = fn(dev); bufts && bufts[0])
                repack_buft = bufts[0];
        }
        if (verbose())
            fprintf(stderr, "[trellis:dbg] extra-bufts: dev=%d reg=%d proc=%d repack=%s\n",
                    dev ? 1 : 0, reg ? 1 : 0, raw ? 1 : 0,
                    repack_buft ? ggml_backend_buft_name(repack_buft) : "(none)");
    }

    // Only matrix weights of types supported by the Hexagon repack buffer are eligible.
    auto repackable = [](const ggml_tensor* t) {
        if (ggml_n_dims(const_cast<ggml_tensor*>(t)) < 2) return false;
        switch (t->type) {
            case GGML_TYPE_Q4_0:
            case GGML_TYPE_Q8_0:
            case GGML_TYPE_MXFP4: return true;
            default:              return false;
        }
    };

    const int64_t n_tensors = gguf_get_n_tensors(m.gguf);
    size_t n_repack = 0;
    if (!repack_buft) {
        // Preserve the existing allocation path for CPU, CUDA, Vulkan, and backends
        // without an extra weight buffer.
        m.buffer = ggml_backend_alloc_ctx_tensors(meta, m.backend);
        if (!m.buffer) throw std::runtime_error("failed to allocate tensor buffer for " + path);
        ggml_backend_buffer_set_usage(m.buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    } else {
        const size_t ctx_size = ggml_tensor_overhead() * (size_t)(n_tensors + 8);
        m.ctx_host   = ggml_init({ ctx_size, nullptr, true });
        m.ctx_repack = ggml_init({ ctx_size, nullptr, true });

        // Re-declare each tensor in the context matching the buffer it should use.
        for (int64_t i = 0; i < n_tensors; ++i) {
            const char*  nm  = gguf_get_tensor_name(m.gguf, i);
            ggml_tensor* src = ggml_get_tensor(meta, nm);
            const bool   rp  = repackable(src);
            ggml_tensor* dst = ggml_dup_tensor(rp ? m.ctx_repack : m.ctx_host, src);
            ggml_set_name(dst, nm);
            if (rp) ++n_repack;
        }

        m.buffer = ggml_backend_alloc_ctx_tensors(m.ctx_host, m.backend);
        if (!m.buffer) throw std::runtime_error("failed to allocate tensor buffer for " + path);
        ggml_backend_buffer_set_usage(m.buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        if (n_repack == 0) {
            ggml_free(m.ctx_repack);
            m.ctx_repack = nullptr;
        } else {
            m.buffer_repack = ggml_backend_alloc_ctx_tensors_from_buft(m.ctx_repack, repack_buft);
            if (!m.buffer_repack) throw std::runtime_error("failed to allocate repack buffer for " + path);
            ggml_backend_buffer_set_usage(m.buffer_repack, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            fprintf(stderr, "[trellis] %s: %zu/%lld tensors -> %s (%.1f MB)\n",
                    path.substr(path.find_last_of("/\\") + 1).c_str(), n_repack, (long long) n_tensors,
                    ggml_backend_buft_name(repack_buft),
                    ggml_backend_buffer_get_size(m.buffer_repack) / 1e6);
        }
    }

    m.sched = make_sched(m.backend);

    if (verbose()) fprintf(stderr, "[trellis:dbg] phase: streaming weights...\n");
    // stream weights from disk into the backend buffer
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot reopen gguf: " + path);
    const size_t data_off = gguf_get_data_offset(m.gguf);
    std::vector<uint8_t> staging;
    const int64_t n = gguf_get_n_tensors(m.gguf);
    for (int64_t i = 0; i < n; ++i) {
        const char* name = gguf_get_tensor_name(m.gguf, i);
        ggml_tensor* t = m.ctx_repack ? ggml_get_tensor(m.ctx_repack, name) : nullptr;
        if (!t && m.ctx_host) t = ggml_get_tensor(m.ctx_host, name);
        if (!t) t = ggml_get_tensor(meta, name);
        const size_t nbytes = ggml_nbytes(t);
        const size_t off = data_off + gguf_get_tensor_offset(m.gguf, i);
        staging.resize(nbytes);
        if (trellis_fseek64(f, (int64_t)off, SEEK_SET) != 0 ||
            fread(staging.data(), 1, nbytes, f) != nbytes) {
            fclose(f);
            throw std::runtime_error(std::string("short read for tensor ") + name);
        }
        ggml_backend_tensor_set(t, staging.data(), 0, nbytes);
        m.tensors[name] = t;
    }
    fclose(f);
    return m;
}

ggml_tensor* Model::try_get(const std::string& name) const {
    auto it = tensors.find(name);
    return it == tensors.end() ? nullptr : it->second;
}

ggml_tensor* Model::get(const std::string& name) const {
    ggml_tensor* t = try_get(name);
    if (!t) throw std::runtime_error("missing tensor: " + name);
    return t;
}

bool Model::has(const std::string& name) const { return tensors.count(name) > 0; }

size_t Model::total_bytes() const {
    size_t s = 0;
    for (auto& [k, t] : tensors) s += ggml_nbytes(t);
    return s;
}

void Model::free() {
    if (sched) ggml_backend_sched_free(sched);
    if (buffer_repack) ggml_backend_buffer_free(buffer_repack);
    if (buffer) ggml_backend_buffer_free(buffer);
    if (ctx_repack) ggml_free(ctx_repack);
    if (ctx_host)   ggml_free(ctx_host);
    // backend is shared process-wide (see shared_backend) -- never freed here.
    if (gguf) gguf_free(gguf);
    if (meta) ggml_free(meta);
    sched = nullptr; buffer = nullptr; buffer_repack = nullptr;
    ctx_host = nullptr; ctx_repack = nullptr;
    backend = nullptr; gguf = nullptr; meta = nullptr;
    tensors.clear();
}

std::vector<float> tensor_to_f32(ggml_tensor* t) {
    const int64_t ne = ggml_nelements(t);
    std::vector<float> out(ne);
    const size_t nbytes = ggml_nbytes(t);
    std::vector<uint8_t> raw(nbytes);
    ggml_backend_tensor_get(t, raw.data(), 0, nbytes);
    if (t->type == GGML_TYPE_F32) {
        memcpy(out.data(), raw.data(), ne * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        ggml_fp16_to_fp32_row((const ggml_fp16_t*)raw.data(), out.data(), ne);
    } else {
        throw std::runtime_error("tensor_to_f32: unsupported type");
    }
    return out;
}

} // namespace trellis
