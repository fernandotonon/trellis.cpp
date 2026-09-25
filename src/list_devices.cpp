// Enumerate registered ggml devices and optionally initialize them.

#include "ggml-backend.h"

#include <cstdio>
#include <cstddef>
#include <string_view>

static const char* dev_type_name(enum ggml_backend_dev_type t) {
    switch (t) {
        case GGML_BACKEND_DEVICE_TYPE_CPU:   return "CPU";
        case GGML_BACKEND_DEVICE_TYPE_GPU:   return "GPU";
        case GGML_BACKEND_DEVICE_TYPE_IGPU:  return "IGPU";
        case GGML_BACKEND_DEVICE_TYPE_ACCEL: return "ACCEL";
        case GGML_BACKEND_DEVICE_TYPE_META:  return "META";
        default:                             return "?";
    }
}

static double to_mib(size_t bytes) {
    return (double) bytes / (1024.0 * 1024.0);
}

int main(int argc, char** argv) {
    const bool probe = (argc > 1 && std::string_view(argv[1]) == "--init");
    bool init_failed = false;

    printf("registered backends: %zu\n", ggml_backend_reg_count());
    for (size_t i = 0; i < ggml_backend_reg_count(); ++i) {
        ggml_backend_reg_t reg = ggml_backend_reg_get(i);
        printf("  [%zu] %-12s  devices: %zu\n", i, ggml_backend_reg_name(reg),
               ggml_backend_reg_dev_count(reg));
    }

    const size_t n = ggml_backend_dev_count();
    printf("\ndevices: %zu\n", n);

    for (size_t i = 0; i < n; ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (!d) {
            printf("\n  device %zu\n    unavailable (backend initialization failed)\n", i);
            init_failed = true;
            continue;
        }

        ggml_backend_dev_props p{};
        ggml_backend_dev_get_props(d, &p);

        printf("\n  device %zu\n", i);
        printf("    name        : %s\n", ggml_backend_dev_name(d));
        printf("    description : %s\n", ggml_backend_dev_description(d));
        printf("    type        : %s\n", dev_type_name(ggml_backend_dev_type(d)));
        printf("    backend reg : %s\n", ggml_backend_reg_name(ggml_backend_dev_backend_reg(d)));
        printf("    memory      : %.0f MiB free / %.0f MiB total\n",
               to_mib(p.memory_free), to_mib(p.memory_total));
        printf("    caps        : async=%d host_buffer=%d from_host_ptr=%d events=%d\n",
               p.caps.async, p.caps.host_buffer, p.caps.buffer_from_host_ptr, p.caps.events);

        // Actually bringing the device up is the real test -- registration alone does
        // not prove the driver/firmware path works (notably for Hexagon, where the HTP
        // skel has to load onto the DSP).
        if (probe) {
            ggml_backend_t b = ggml_backend_dev_init(d, nullptr);
            if (b) {
                printf("    init        : OK (%s)\n", ggml_backend_name(b));
                ggml_backend_free(b);
            } else {
                printf("    init        : FAILED\n");
                init_failed = true;
            }
        }
    }

    if (!probe) {
        printf("\n(pass --init to also try bringing each device up)\n");
    }
    return init_failed ? 1 : 0;
}
