#include "quad_topology.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace trellis {
namespace {
struct Edge {
    uint32_t a, b, corner_a, corner_b, face;
    bool direction;
};

uint32_t root(std::vector<uint32_t>& parent, uint32_t value) {
    while (parent[value] != value) {
        parent[value] = parent[parent[value]];
        value = parent[value];
    }
    return value;
}

void join(std::vector<uint32_t>& parent, uint32_t a, uint32_t b) {
    a = root(parent, a);
    b = root(parent, b);
    if (a != b) parent[std::max(a, b)] = std::min(a, b);
}
}

QuadTopologyAudit audit_quad_topology(
    size_t vertex_count, const std::vector<std::array<uint32_t, 4>>& quads) {
    if (!vertex_count || quads.empty() ||
        vertex_count > std::numeric_limits<uint32_t>::max() ||
        quads.size() > std::numeric_limits<uint32_t>::max() / 4)
        throw std::invalid_argument("Invalid quad topology input size");
    QuadTopologyAudit result;
    result.vertices = vertex_count;
    result.quads = quads.size();
    std::vector<Edge> edges;
    edges.reserve(quads.size() * 4);
    std::vector<std::array<uint32_t,4>> canonical;
    canonical.reserve(quads.size());
    for (uint32_t face = 0; face < quads.size(); ++face) {
        const auto& q = quads[face];
        auto sorted = q;
        std::sort(sorted.begin(), sorted.end());
        if (sorted.back() >= vertex_count ||
            std::adjacent_find(sorted.begin(),sorted.end()) != sorted.end())
            throw std::invalid_argument("Invalid quad vertex indices");
        canonical.push_back(sorted);
        for (uint32_t j = 0; j < 4; ++j) {
            const uint32_t k = (j + 1) % 4;
            const bool direction = q[j] < q[k];
            edges.push_back({std::min(q[j],q[k]), std::max(q[j],q[k]),
                             4 * face + (direction ? j : k),
                             4 * face + (direction ? k : j), face, direction});
        }
    }
    std::sort(canonical.begin(), canonical.end());
    for (size_t i = 1; i < canonical.size(); ++i)
        result.duplicate_faces += canonical[i] == canonical[i-1];
    std::sort(edges.begin(), edges.end(), [](const Edge& x, const Edge& y) {
        return x.a != y.a ? x.a < y.a : x.b < y.b;
    });
    std::vector<uint32_t> face_parent(quads.size()), corner_parent(quads.size() * 4);
    std::iota(face_parent.begin(),face_parent.end(),0);
    std::iota(corner_parent.begin(),corner_parent.end(),0);
    for (size_t i = 0; i < edges.size();) {
        size_t end = i + 1;
        while (end < edges.size() && edges[end].a == edges[i].a &&
               edges[end].b == edges[i].b) ++end;
        if (end - i == 1) ++result.boundary_edges;
        else {
            if (end - i > 2) ++result.nonmanifold_edges;
            if (end - i == 2 && edges[i].direction == edges[i+1].direction)
                ++result.winding_conflicts;
            for (size_t j = i + 1; j < end; ++j) {
                join(face_parent,edges[i].face,edges[j].face);
                join(corner_parent,edges[i].corner_a,edges[j].corner_a);
                join(corner_parent,edges[i].corner_b,edges[j].corner_b);
            }
        }
        i = end;
    }
    std::vector<uint8_t> component_seen(quads.size());
    for (uint32_t face = 0; face < quads.size(); ++face) {
        const uint32_t label = root(face_parent,face);
        if (!component_seen[label]) {
            component_seen[label] = 1;
            ++result.components;
        }
    }
    std::vector<std::pair<uint32_t,uint32_t>> fans;
    fans.reserve(quads.size() * 4);
    for (uint32_t face = 0; face < quads.size(); ++face)
        for (uint32_t j = 0; j < 4; ++j)
            fans.push_back({quads[face][j],root(corner_parent,4 * face + j)});
    std::sort(fans.begin(),fans.end());
    uint32_t previous_vertex = std::numeric_limits<uint32_t>::max();
    uint32_t previous_fan = std::numeric_limits<uint32_t>::max();
    bool split_vertex = false;
    for (const auto& [vertex,fan] : fans) {
        if (vertex != previous_vertex) split_vertex = false;
        else if (fan != previous_fan && !split_vertex) {
            ++result.nonmanifold_vertices;
            split_vertex = true;
        }
        previous_vertex = vertex;
        previous_fan = fan;
    }
    return result;
}
}
