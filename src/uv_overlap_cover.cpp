#include "uv_overlap_cover.h"
#include <algorithm>
#include <queue>
#include <stdexcept>

namespace trellis {

std::vector<uint32_t> greedy_uv_overlap_cover(
    size_t face_count, const std::vector<std::array<uint32_t, 2>>& overlaps,
    const std::vector<uint32_t>& required_faces) {
    if (face_count > UINT32_MAX || overlaps.size() > UINT32_MAX)
        throw std::invalid_argument("Too many UV faces or overlap pairs");
    std::vector<std::vector<uint32_t>> adjacency(face_count);
    std::vector<uint32_t> degree(face_count, 0);
    std::vector<uint8_t> chosen(face_count, 0), active(overlaps.size(), 1);
    for (auto face : required_faces) {
        if (face >= face_count) throw std::invalid_argument("Invalid required UV face");
        chosen[face] = 1;
    }
    for (uint32_t edge = 0; edge < overlaps.size(); ++edge) {
        const auto [a, b] = overlaps[edge];
        if (a >= face_count || b >= face_count || a == b)
            throw std::invalid_argument("Invalid UV overlap pair");
        adjacency[a].push_back(edge);
        adjacency[b].push_back(edge);
        if (chosen[a] || chosen[b]) active[edge] = 0;
        else { ++degree[a]; ++degree[b]; }
    }
    struct Node { uint32_t degree, face; };
    auto lower = [](Node a, Node b) {
        return a.degree != b.degree ? a.degree < b.degree : a.face > b.face;
    };
    std::priority_queue<Node, std::vector<Node>, decltype(lower)> heap(lower);
    for (uint32_t face = 0; face < face_count; ++face)
        if (degree[face]) heap.push({degree[face], face});
    while (!heap.empty()) {
        const auto node = heap.top(); heap.pop();
        if (node.degree != degree[node.face] || !node.degree) continue;
        chosen[node.face] = 1;
        for (auto edge : adjacency[node.face]) {
            if (!active[edge]) continue;
            active[edge] = 0;
            const auto [a, b] = overlaps[edge];
            const auto other = a == node.face ? b : a;
            --degree[other];
            if (degree[other]) heap.push({degree[other], other});
        }
        degree[node.face] = 0;
    }
    std::vector<uint32_t> result;
    for (uint32_t face = 0; face < face_count; ++face)
        if (chosen[face]) result.push_back(face);
    return result;
}

}
