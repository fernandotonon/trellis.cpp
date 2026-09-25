#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <set>
#include <vector>

namespace trellis {

struct QuadMatchOptions {
    double minimum_scaled_jacobian = 0.1;
    double maximum_warpage = std::numeric_limits<double>::infinity();
    double minimum_corner_degrees = 0.0;
    double maximum_normal_span_degrees = 180.0;
    bool weighted = true;
    bool corner_objective = false;
    std::set<std::array<size_t, 2>> forbidden_pairs;
    std::set<size_t> preferred_faces;
    std::vector<double> face_warpage;
};

struct QuadMatchResult {
    std::vector<std::array<uint32_t, 4>> quads;
    std::vector<std::array<size_t, 2>> quad_source_faces;
    std::vector<double> quad_quality;
    std::vector<size_t> unmatched_faces;
    std::vector<int> eligible_degree;
    std::vector<double> best_quality;
    size_t candidate_count = 0;
    size_t eligible_count = 0;
};

QuadMatchResult match_triangles_to_quads(
    const std::vector<std::array<float, 3>>& vertices,
    const std::vector<std::array<uint32_t, 3>>& triangles,
    const QuadMatchOptions& options);

}  // namespace trellis
