#include "quad_collision.h"

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polygon_mesh_processing/self_intersections.h>

#include <algorithm>
#include <map>
#include <stdexcept>
#include <sstream>
#include <utility>

namespace trellis {

QuadCollisionReport audit_quad_collisions(
    const std::vector<std::array<float, 3>>& vertices,
    const std::vector<std::array<uint32_t, 3>>& source_triangles,
    const QuadMatchResult& match) {
    using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
    using Edge = std::array<uint32_t, 2>;
    if (match.quads.size() != match.quad_source_faces.size())
        throw std::invalid_argument("quad provenance size mismatch");
    std::vector<Kernel::Point_3> points;
    points.reserve(vertices.size());
    for (const auto& p : vertices) points.emplace_back(p[0], p[1], p[2]);
    std::vector<std::array<size_t, 3>> triangles;
    std::vector<size_t> owners;
    triangles.reserve(match.quads.size()*4 + match.unmatched_faces.size());
    owners.reserve(triangles.capacity());
    std::map<Edge, std::set<size_t>> boundaries, diagonals;
    const auto edge = [](uint32_t a, uint32_t b) -> Edge {
        return {std::min(a,b), std::max(a,b)};
    };
    const auto add_triangle = [&](const std::array<uint32_t, 3>& t, size_t owner) {
        for (uint32_t id : t)
            if (id >= vertices.size()) throw std::invalid_argument("quad vertex out of range");
        triangles.push_back({t[0], t[1], t[2]});
        owners.push_back(owner);
    };
    for (size_t i = 0; i < match.quads.size(); ++i) {
        const auto& q = match.quads[i];
        for (int j = 0; j < 4; ++j) boundaries[edge(q[j], q[(j+1)%4])].insert(i);
        diagonals[edge(q[0],q[2])].insert(i);
        diagonals[edge(q[1],q[3])].insert(i);
        add_triangle({q[0],q[1],q[2]}, i);
        add_triangle({q[0],q[2],q[3]}, i);
        add_triangle({q[1],q[2],q[3]}, i);
        add_triangle({q[1],q[3],q[0]}, i);
    }
    for (size_t i = 0; i < match.unmatched_faces.size(); ++i) {
        const size_t face = match.unmatched_faces[i];
        if (face >= source_triangles.size()) throw std::invalid_argument("source face out of range");
        const auto& t = source_triangles[face];
        const size_t owner = match.quads.size()+i;
        for (int j = 0; j < 3; ++j) boundaries[edge(t[j],t[(j+1)%3])].insert(owner);
        add_triangle(t, owner);
    }

    std::vector<std::pair<size_t, size_t>> pairs;
    CGAL::Polygon_mesh_processing::triangle_soup_self_intersections(
        points, triangles, std::back_inserter(pairs));
    QuadCollisionReport report;
    for (const auto [a,b] : pairs) {
        if (a == b) {
            ++report.degenerate_triangles;
            report.affected_polygons.insert(owners[a]);
        } else if (owners[a] != owners[b]) {
            ++report.cross_polygon_intersections;
            report.affected_polygons.insert(owners[a]);
            report.affected_polygons.insert(owners[b]);
        }
    }
    std::map<std::array<size_t,3>, std::vector<size_t>> duplicates;
    for (size_t i = 0; i < triangles.size(); ++i) {
        auto key = triangles[i];
        std::sort(key.begin(), key.end());
        for (size_t previous : duplicates[key])
            if (owners[previous] != owners[i]) {
                ++report.duplicate_triangle_pairs;
                report.affected_polygons.insert(owners[previous]);
                report.affected_polygons.insert(owners[i]);
            }
        duplicates[key].push_back(i);
    }
    for (const auto& [key, diagonal_owners] : diagonals) {
        std::set<size_t> involved = diagonal_owners;
        const auto it = boundaries.find(key);
        if (it != boundaries.end()) involved.insert(it->second.begin(), it->second.end());
        if (involved.size() > 1) {
            ++report.diagonal_edge_conflicts;
            report.affected_polygons.insert(involved.begin(), involved.end());
        }
    }
    return report;
}

QuadCollisionReport audit_quad_collisions(
    const std::vector<std::array<float, 3>>& vertices,
    const std::vector<std::array<uint32_t, 4>>& quads) {
    QuadMatchResult mesh;
    mesh.quads = quads;
    mesh.quad_source_faces.resize(quads.size());
    return audit_quad_collisions(vertices, {}, mesh);
}

SafeQuadMatchResult match_triangles_to_quads_collision_safe(
    const std::vector<std::array<float, 3>>& vertices,
    const std::vector<std::array<uint32_t, 3>>& source_triangles,
    QuadMatchOptions options,
    size_t maximum_rounds) {
    if (!maximum_rounds) throw std::invalid_argument("zero matching rounds");
    SafeQuadMatchResult result;
    for (size_t round = 0; round < maximum_rounds; ++round) {
        result.match = match_triangles_to_quads(vertices, source_triangles, options);
        auto collision = audit_quad_collisions(vertices, source_triangles, result.match);
        size_t added = 0;
        if (!collision.clean())
            for (size_t polygon : collision.affected_polygons)
                if (polygon < result.match.quads.size()) {
                    auto pair = result.match.quad_source_faces[polygon];
                    if (pair[0] > pair[1]) std::swap(pair[0], pair[1]);
                    added += options.forbidden_pairs.insert(pair).second;
                }
        result.rounds.push_back({std::move(collision), options.forbidden_pairs.size(), added,
                                 result.match.quads.size(), result.match.unmatched_faces.size()});
        if (result.rounds.back().collision.clean()) return result;
        if (!added) {
            size_t affected_quads=0;
            for (size_t polygon:result.rounds.back().collision.affected_polygons)
                affected_quads+=polygon<result.match.quads.size();
            std::ostringstream detail;
            detail<<"no removable quad explains collision defects: round="<<round
                  <<" crossings="<<result.rounds.back().collision.cross_polygon_intersections
                  <<" duplicates="<<result.rounds.back().collision.duplicate_triangle_pairs
                  <<" chords="<<result.rounds.back().collision.diagonal_edge_conflicts
                  <<" degeneracies="<<result.rounds.back().collision.degenerate_triangles
                  <<" affected_quads="<<affected_quads
                  <<" affected_residual_triangles="
                  <<result.rounds.back().collision.affected_polygons.size()-affected_quads;
            throw std::runtime_error(detail.str());
        }
    }
    throw std::runtime_error("collision-safe matching did not converge");
}

}
