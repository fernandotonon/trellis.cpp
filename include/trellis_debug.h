#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <chrono>
#include <string>

namespace trellis {

// --verbose / TRELLIS_VERBOSE=1 enables graph timings and progress heartbeats.
bool verbose();
void set_verbose(bool v);

// Scoped wall-clock timer. Logs "<label>: <n> ms" on destruction when verbose().
class ScopedTimer {
public:
    explicit ScopedTimer(std::string label);
    ~ScopedTimer();
    double ms() const;

private:
    std::string                                    label_;
    std::chrono::steady_clock::time_point          t0_;
};

// Instrumented graph execution for a single backend or scheduler.
ggml_status compute_graph(ggml_backend_t backend, ggml_cgraph* graph, const char* label);
ggml_status compute_graph_sched(ggml_backend_sched_t sched, ggml_cgraph* graph, const char* label);

// Log a graph's shape (node/leaf counts, and the op histogram at verbose level 2).
void log_graph(const char* label, const ggml_cgraph* graph);

}  // namespace trellis
