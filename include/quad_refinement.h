#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <set>
#include <utility>
#include <vector>

namespace trellis {

using QuadRefinementEdge = std::array<size_t, 2>;

struct QuadRefinementOptions {
    bool surface = true;
    double center_normal_span_degrees = 180.0;
    std::set<size_t> forced_faces;
    std::set<size_t> frozen_faces;
    bool use_split_seed = false;
    std::set<QuadRefinementEdge> split_seed;
};

struct QuadRefinementResult {
    std::vector<std::array<float, 3>> vertices;
    std::vector<std::array<uint32_t, 4>> quads;
    std::vector<size_t> parents;
    std::vector<double> quality;
    std::vector<std::vector<std::pair<size_t, double>>> vertex_witnesses;
    std::set<QuadRefinementEdge> split_edges;
    size_t parity_paths = 0;
    size_t unchanged = 0;
    size_t opposite = 0;
    size_t adjacent = 0;
    size_t full = 0;
    size_t optimized_centers = 0;
    size_t rotated_for_source_coverage = 0;
    size_t unproven_quads = 0;
};

QuadRefinementResult refine_quads(
    const std::vector<std::array<float, 3>>& vertices,
    const std::vector<std::vector<uint32_t>>& faces,
    const QuadRefinementOptions& options);

struct QuadRefinementGateReport {
    std::set<size_t> bad_quality_parents;
    std::set<size_t> bad_shape_parents;
    std::set<size_t> unproven_parents;
    size_t nonpositive_witness_triangles = 0;
};

QuadRefinementGateReport audit_quad_refinement(
    const std::vector<std::vector<uint32_t>>& source_faces,
    const QuadRefinementResult& result);

struct QuadRefinementRound {
    size_t forced_parents = 0;
    size_t new_forced_parents = 0;
    size_t bad_quality_parents = 0;
    size_t bad_shape_parents = 0;
    size_t unproven_parents = 0;
    size_t quad_faces = 0;
};

struct QualifiedQuadRefinementResult {
    QuadRefinementResult mesh;
    std::vector<QuadRefinementRound> rounds;
    bool gates_passed = true;
};

QualifiedQuadRefinementResult refine_quads_until_qualified(
    const std::vector<std::array<float, 3>>& vertices,
    const std::vector<std::vector<uint32_t>>& faces,
    QuadRefinementOptions options,
    size_t maximum_rounds = 35,
    bool permit_shape_repair = false);

double quad_corner_normal_span_degrees(
    const std::array<std::array<float, 3>, 4>& points);

}
