#include "quad_components.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace trellis {

QuadComponentLabels label_quad_components(
    size_t vertex_count, const std::vector<std::array<uint32_t, 4>>& quads) {
    if (!vertex_count || quads.empty()) throw std::invalid_argument("Empty quad mesh");
    struct Edge { uint32_t a, b; size_t face; };
    std::vector<Edge> edges;
    edges.reserve(quads.size()*4);
    for (size_t qi = 0; qi < quads.size(); ++qi) {
        const auto& q = quads[qi];
        for (uint32_t vertex : q)
            if (vertex >= vertex_count) throw std::invalid_argument("Quad vertex out of range");
        auto unique = q;
        std::sort(unique.begin(),unique.end());
        if (std::adjacent_find(unique.begin(),unique.end()) != unique.end())
            throw std::invalid_argument("Degenerate quad index");
        for (int j = 0; j < 4; ++j)
            edges.push_back({std::min(q[j],q[(j+1)%4]),
                             std::max(q[j],q[(j+1)%4]),qi});
    }
    std::sort(edges.begin(),edges.end(),[](const Edge& a, const Edge& b) {
        return a.a != b.a ? a.a < b.a : a.b < b.b;
    });
    std::vector<size_t> parent(quads.size());
    std::iota(parent.begin(),parent.end(),0);
    const auto root = [&](size_t face) {
        while (parent[face] != face) {
            parent[face] = parent[parent[face]];
            face = parent[face];
        }
        return face;
    };
    for (size_t i = 1; i < edges.size(); ++i)
        if (edges[i].a == edges[i-1].a && edges[i].b == edges[i-1].b) {
            const size_t a = root(edges[i].face), b = root(edges[i-1].face);
            parent[std::max(a,b)] = std::min(a,b);
        }
    QuadComponentLabels result;
    result.face_labels.resize(quads.size());
    std::vector<size_t> root_labels(quads.size(),quads.size());
    for (size_t qi = 0; qi < quads.size(); ++qi) {
        const size_t r = root(qi);
        if (root_labels[r] == quads.size()) {
            root_labels[r] = result.face_counts.size();
            result.face_counts.push_back(0);
        }
        result.face_labels[qi] = root_labels[r];
        ++result.face_counts[root_labels[r]];
    }
    return result;
}

FilteredQuadComponents filter_quad_components(
    const std::vector<std::array<float, 3>>& vertices,
    const std::vector<std::array<uint32_t, 4>>& quads,
    const QuadComponentLabels& labels,
    const std::set<size_t>& keep) {
    if (labels.face_labels.size() != quads.size() || keep.empty())
        throw std::invalid_argument("Invalid quad component selection");
    for (size_t id : keep)
        if (id >= labels.face_counts.size())
            throw std::invalid_argument("Component ID out of range");
    FilteredQuadComponents result;
    std::vector<bool> used(vertices.size(),false);
    for (size_t qi = 0; qi < quads.size(); ++qi)
        if (keep.count(labels.face_labels[qi])) {
            result.source_face_ids.push_back(qi);
            for (uint32_t vertex : quads[qi]) {
                if (vertex >= vertices.size())
                    throw std::invalid_argument("Quad vertex out of range");
                used[vertex] = true;
            }
        }
    if (result.source_face_ids.empty())
        throw std::invalid_argument("No quads in selected components");
    std::vector<uint32_t> remap(vertices.size(),UINT32_MAX);
    for (size_t vertex = 0; vertex < vertices.size(); ++vertex)
        if (used[vertex]) {
            remap[vertex] = uint32_t(result.vertices.size());
            result.vertices.push_back(vertices[vertex]);
            result.source_vertex_ids.push_back(uint32_t(vertex));
        }
    for (size_t qi : result.source_face_ids) {
        auto quad = quads[qi];
        for (uint32_t& vertex : quad) vertex = remap[vertex];
        result.quads.push_back(quad);
    }
    return result;
}

}
