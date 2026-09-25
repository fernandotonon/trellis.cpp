#pragma once

#include "quad_collision.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace trellis {

struct QuadLocalRepairOptions {
    double neighborhood_padding = 0.008;
    double maximum_vertex_move = 0.00005;
    double maximum_incident_edge_fraction = 0.02;
    size_t maximum_steps_per_group = 20;
};

struct QuadLocalRepairMove {
    uint32_t vertex = 0;
    std::array<float, 3> before{};
    std::array<float, 3> after{};
    size_t remaining_crossings = 0;
};

struct QuadLocalRepairResult {
    QuadCollisionReport before;
    QuadCollisionReport after;
    std::vector<QuadLocalRepairMove> moves;
};

struct QuadShapeRepairResult {
    size_t bad_quads_before = 0;
    size_t bad_quads_after = 0;
    std::vector<QuadLocalRepairMove> moves;
};

QuadShapeRepairResult repair_quad_shapes(
    std::vector<std::array<float, 3>>& vertices,
    const std::vector<std::array<uint32_t, 4>>& quads,
    const QuadLocalRepairOptions& options = {});

using QuadSourceProjection =
    std::function<std::array<float, 3>(const std::array<float, 3>&)>;

QuadLocalRepairResult repair_quad_collisions(
    std::vector<std::array<float, 3>>& vertices,
    const std::vector<std::array<uint32_t, 4>>& quads,
    const QuadSourceProjection& project_to_source,
    const QuadLocalRepairOptions& options = {});

}
