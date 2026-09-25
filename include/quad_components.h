#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <set>
#include <vector>

namespace trellis {

struct QuadComponentLabels {
    std::vector<size_t> face_labels;
    std::vector<size_t> face_counts;
};

QuadComponentLabels label_quad_components(
    size_t vertex_count,
    const std::vector<std::array<uint32_t, 4>>& quads);

struct FilteredQuadComponents {
    std::vector<std::array<float, 3>> vertices;
    std::vector<std::array<uint32_t, 4>> quads;
    std::vector<uint32_t> source_vertex_ids;
    std::vector<size_t> source_face_ids;
};

FilteredQuadComponents filter_quad_components(
    const std::vector<std::array<float, 3>>& vertices,
    const std::vector<std::array<uint32_t, 4>>& quads,
    const QuadComponentLabels& labels,
    const std::set<size_t>& keep);

struct QuadSourceCover {
    std::set<size_t> keep;
    size_t rounds = 0;
    double maximum_leaf_upper_bound = 0;
    double maximum_reverse_leaf_upper_bound = 0;
    uint64_t distance_queries = 0;
    uint64_t reverse_distance_queries = 0;
};

QuadSourceCover select_source_covering_quad_components(
    const std::vector<std::array<float, 3>>& source_vertices,
    const std::vector<std::array<int32_t, 3>>& source_faces,
    const std::vector<std::array<float, 3>>& quad_vertices,
    const std::vector<std::array<uint32_t, 4>>& quads,
    const QuadComponentLabels& labels,
    double limit,
    int max_depth = 12);

}
