#include "quad_quality.h"

#include <array>
#include <cmath>
#include <cstdio>

int main() {
    using trellis::QuadPoint;
    using Quad = std::array<QuadPoint, 4>;
    const Quad quads[] = {
        {{{0,0,0}, {1,0,0}, {1,1,0}, {0,1,0}}},
        {{{0,0,0}, {2,0,0}, {2,1,0}, {0,1,0}}},
        {{{0,0,0}, {1,0,0}, {1,.1,0}, {0,1,0}}},
        {{{0,0,0}, {1,0,0}, {.3,.1,0}, {0,1,0}}},
        {{{0,0,0}, {1,0,0}, {1,1,.5}, {0,1,0}}},
        {{{0,0,0}, {1,0,0}, {1,1,2}, {0,1,0}}},
        {{{0,0,0}, {1,0,0}, {.2,.2,1}, {0,1,0}}},
    };
    const double expected[] = {1, 1, .18181818428118368, -2.9999998323619419,
                               .88888888888888906, .33333333333333343,
                               .3703703750716969};
    for (size_t i = 0; i < std::size(quads); ++i) {
        const double actual = trellis::quad_min_scaled_jacobian(quads[i]);
        if (std::abs(actual - expected[i]) > 1e-6) {
            std::fprintf(stderr, "quad %zu: expected %.9g, got %.9g\n", i, expected[i], actual);
            return 1;
        }
    }
    const Quad degenerate = {{{0,0,0}, {1,0,0}, {0,0,0}, {1,0,0}}};
    if (trellis::quad_min_scaled_jacobian(degenerate) != -INFINITY) return 1;
    return 0;
}
