#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace trellis {

using QuadPoint = std::array<double, 3>;

inline double quad_min_scaled_jacobian(const std::array<QuadPoint, 4>& p) {
    auto sub = [](const QuadPoint& a, const QuadPoint& b) {
        return QuadPoint{a[0] - b[0], a[1] - b[1], a[2] - b[2]};
    };
    auto cross = [](const QuadPoint& a, const QuadPoint& b) {
        return QuadPoint{a[1] * b[2] - a[2] * b[1],
                         a[2] * b[0] - a[0] * b[2],
                         a[0] * b[1] - a[1] * b[0]};
    };
    auto dot = [](const QuadPoint& a, const QuadPoint& b) {
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    };
    const QuadPoint u0 = sub(p[1], p[0]);
    const QuadPoint u1 = sub(p[2], p[3]);
    const QuadPoint v0 = sub(p[3], p[0]);
    const QuadPoint v1 = sub(p[2], p[1]);
    const QuadPoint du = {(u0[0] + u1[0]) * 0.5, (u0[1] + u1[1]) * 0.5,
                          (u0[2] + u1[2]) * 0.5};
    const QuadPoint dv = {(v0[0] + v1[0]) * 0.5, (v0[1] + v1[1]) * 0.5,
                          (v0[2] + v1[2]) * 0.5};
    const QuadPoint center = cross(du, dv);
    const double denominator = dot(center, center);
    if (!(denominator > 0.0) || !std::isfinite(denominator))
        return -std::numeric_limits<double>::infinity();
    const double corners[] = {
        dot(cross(u0, v0), center), dot(cross(u0, v1), center),
        dot(cross(u1, v1), center), dot(cross(u1, v0), center),
    };
    double minimum = corners[0];
    for (double value : corners) {
        if (!std::isfinite(value)) return -std::numeric_limits<double>::infinity();
        minimum = std::min(minimum, value);
    }
    return minimum / denominator;
}

}  // namespace trellis
