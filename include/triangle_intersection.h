#pragma once
#include <algorithm>
#include <array>
#include <cmath>

namespace trellis {

using Triangle3 = std::array<std::array<double, 3>, 3>;

inline bool triangles_overlap_interior(const Triangle3& left, const Triangle3& right) {
    using V = std::array<double, 3>;
    auto sub = [](const V& a, const V& b) -> V {
        return {a[0]-b[0], a[1]-b[1], a[2]-b[2]};
    };
    auto cross = [](const V& a, const V& b) -> V {
        return {a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]};
    };
    auto separated = [&](const V& axis) {
        if (axis[0] == 0.0 && axis[1] == 0.0 && axis[2] == 0.0) return false;
        double a0 = 1e300, a1 = -1e300, b0 = 1e300, b1 = -1e300;
        for (int i = 0; i < 3; ++i) {
            const double a = left[i][0]*axis[0] + left[i][1]*axis[1] + left[i][2]*axis[2];
            const double b = right[i][0]*axis[0] + right[i][1]*axis[1] + right[i][2]*axis[2];
            a0 = std::min(a0, a); a1 = std::max(a1, a);
            b0 = std::min(b0, b); b1 = std::max(b1, b);
        }
        return a1 < b0 || b1 < a0;
    };
    V ae[3], be[3];
    for (int i = 0; i < 3; ++i) {
        ae[i] = sub(left[(i+1)%3], left[i]);
        be[i] = sub(right[(i+1)%3], right[i]);
    }
    const V an = cross(ae[0], ae[1]), bn = cross(be[0], be[1]);
    if (separated(an) || separated(bn)) return false;
    for (const V& a : ae)
        for (const V& b : be)
            if (separated(cross(a,b))) return false;
    for (const V& e : ae) if (separated(cross(an,e))) return false;
    for (const V& e : be) if (separated(cross(bn,e))) return false;
    return true;
}

inline Triangle3 inset_triangle(const Triangle3& triangle, double fraction = 1e-7) {
    Triangle3 result = triangle;
    std::array<double,3> center{};
    for (const auto& point : triangle)
        for (int k = 0; k < 3; ++k) center[k] += point[k] / 3.0;
    for (auto& point : result)
        for (int k = 0; k < 3; ++k) point[k] += fraction * (center[k] - point[k]);
    return result;
}

}  // namespace trellis
