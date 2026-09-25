#pragma once

#include "quad_matching.h"

#include <cstddef>
#include <set>
#include <vector>

namespace trellis {

struct QuadCollisionReport {
    size_t cross_polygon_intersections = 0;
    size_t degenerate_triangles = 0;
    size_t duplicate_triangle_pairs = 0;
    size_t diagonal_edge_conflicts = 0;
    std::set<size_t> affected_polygons;

    bool clean() const {
        return cross_polygon_intersections == 0 && degenerate_triangles == 0 &&
               duplicate_triangle_pairs == 0 && diagonal_edge_conflicts == 0;
    }
};

QuadCollisionReport audit_quad_collisions(
    const std::vector<std::array<float, 3>>& vertices,
    const std::vector<std::array<uint32_t, 3>>& source_triangles,
    const QuadMatchResult& match);

QuadCollisionReport audit_quad_collisions(
    const std::vector<std::array<float, 3>>& vertices,
    const std::vector<std::array<uint32_t, 4>>& quads);

struct SafeQuadMatchRound {
    QuadCollisionReport collision;
    size_t forbidden_pairs = 0;
    size_t new_forbidden_pairs = 0;
    size_t quads = 0;
    size_t residual_triangles = 0;
};

struct SafeQuadMatchResult {
    QuadMatchResult match;
    std::vector<SafeQuadMatchRound> rounds;
};

SafeQuadMatchResult match_triangles_to_quads_collision_safe(
    const std::vector<std::array<float, 3>>& vertices,
    const std::vector<std::array<uint32_t, 3>>& source_triangles,
    QuadMatchOptions options,
    size_t maximum_rounds = 20);

}
