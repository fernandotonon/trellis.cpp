#include "trellis_debug.h"

#include <functional>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <thread>

namespace trellis {

namespace {
int  g_verbose = -1;          // -1 = not yet resolved from the environment
}  // namespace

bool verbose() {
    if (g_verbose < 0) {
        const char* e = std::getenv("TRELLIS_VERBOSE");
        g_verbose = (e && atoi(e) > 0) ? atoi(e) : 0;
    }
    return g_verbose > 0;
}

void set_verbose(bool v) {
    if (v) { if (g_verbose < 1) g_verbose = 1; }
    else   { if (g_verbose < 0) g_verbose = 0; }
}

static int verbose_level() { verbose(); return g_verbose < 0 ? 0 : g_verbose; }

ScopedTimer::ScopedTimer(std::string label)
    : label_(std::move(label)), t0_(std::chrono::steady_clock::now()) {}

double ScopedTimer::ms() const {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0_).count();
}

ScopedTimer::~ScopedTimer() {
    if (verbose()) fprintf(stderr, "[trellis:dbg] %-28s %10.1f ms\n", label_.c_str(), ms());
}

void log_graph(const char* label, const ggml_cgraph* graph) {
    if (!verbose() || !graph) return;
    const int n = ggml_graph_n_nodes(const_cast<ggml_cgraph*>(graph));
    fprintf(stderr, "[trellis:dbg] %-28s %d nodes\n", label, n);

    if (verbose_level() < 2) return;
    std::map<std::string, int> hist;
    for (int i = 0; i < n; ++i) {
        ggml_tensor* t = ggml_graph_node(const_cast<ggml_cgraph*>(graph), i);
        hist[ggml_op_name(t->op)]++;
    }
    for (const auto& [op, count] : hist)
        fprintf(stderr, "[trellis:dbg]     %-24s %5d\n", op.c_str(), count);
}

// Shared instrumentation for potentially long-running graph execution.
static ggml_status run_instrumented(const char* label, const char* target, int n_nodes,
                                    const std::function<ggml_status()>& run,
                                    const std::function<void()>& report_extra) {
    if (!verbose()) return run();

    fprintf(stderr, "[trellis:dbg] %-28s -> %s, %d nodes ...\n", label, target, n_nodes);
    fflush(stderr);

    std::mutex              mu;
    std::condition_variable cv;
    bool                    done = false;
    const auto              t0 = std::chrono::steady_clock::now();

    std::thread hb([&] {
        std::unique_lock<std::mutex> lk(mu);
        while (!cv.wait_for(lk, std::chrono::seconds(10), [&] { return done; })) {
            const double el =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            fprintf(stderr, "[trellis:dbg] %-28s still running, %.0fs\n", label, el);
            fflush(stderr);
        }
    });

    const ggml_status st = run();

    { std::lock_guard<std::mutex> lk(mu); done = true; }
    cv.notify_all();
    hb.join();

    const double el = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0).count();
    fprintf(stderr, "[trellis:dbg] %-28s done in %.1f ms%s\n", label, el,
            st == GGML_STATUS_SUCCESS ? "" : "  (FAILED)");
    if (report_extra) report_extra();
    fflush(stderr);
    return st;
}

ggml_status compute_graph(ggml_backend_t backend, ggml_cgraph* graph, const char* label) {
    const int n = graph ? ggml_graph_n_nodes(graph) : 0;
    return run_instrumented(label, backend ? ggml_backend_name(backend) : "?", n,
                            [&] { return ggml_backend_graph_compute(backend, graph); },
                            nullptr);
}

ggml_status compute_graph_sched(ggml_backend_sched_t sched, ggml_cgraph* graph, const char* label) {
    const int n = graph ? ggml_graph_n_nodes(graph) : 0;
    return run_instrumented(
        label, "sched", n,
        [&] { return ggml_backend_sched_graph_compute(sched, graph); },
        [&] {
            // Splits are the cost of heterogeneity: each one is a handoff (and possibly a
            // tensor copy) between backends. One split = the graph stayed on one device.
            fprintf(stderr, "[trellis:dbg] %-28s %d splits, %d copies\n", label,
                    ggml_backend_sched_get_n_splits(sched),
                    ggml_backend_sched_get_n_copies(sched));
        });
}

}  // namespace trellis
