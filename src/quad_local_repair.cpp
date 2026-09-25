#include "quad_local_repair.h"
#include "quad_quality.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>

namespace trellis {
namespace {

using Point = std::array<float, 3>;
using Quad = std::array<uint32_t, 4>;

double distance(const Point& a, const Point& b) {
    double squared = 0;
    for (int k = 0; k < 3; ++k) {
        const double d = double(a[k]) - b[k];
        squared += d*d;
    }
    return std::sqrt(squared);
}

double shape_excess(const std::vector<Point>& vertices, const Quad& quad) {
    std::array<QuadPoint, 4> points;
    for (int j = 0; j < 4; ++j) {
        if (quad[j] >= vertices.size()) return INFINITY;
        for (int k = 0; k < 3; ++k) points[j][k] = vertices[quad[j]][k];
    }
    double excess = std::max(0.0, 0.1 - quad_min_scaled_jacobian(points));
    std::array<QuadPoint, 4> normals;
    double shortest = std::numeric_limits<double>::infinity(), longest = 0;
    for (int j = 0; j < 4; ++j) {
        QuadPoint next{}, previous{};
        double n2 = 0, p2 = 0, dot = 0;
        for (int k = 0; k < 3; ++k) {
            next[k] = points[(j+1)%4][k] - points[j][k];
            previous[k] = points[(j+3)%4][k] - points[j][k];
            n2 += next[k]*next[k];
            p2 += previous[k]*previous[k];
            dot += next[k]*previous[k];
        }
        if (!(n2 > 0 && p2 > 0)) return INFINITY;
        const double edge_length = std::sqrt(n2);
        shortest = std::min(shortest, edge_length);
        longest = std::max(longest, edge_length);
        const double angle = std::acos(std::clamp(dot/std::sqrt(n2*p2), -1.0, 1.0));
        excess += std::max(0.0, 1 - angle*180/3.141592653589793);
        normals[j] = {next[1]*previous[2]-next[2]*previous[1],
                      next[2]*previous[0]-next[0]*previous[2],
                      next[0]*previous[1]-next[1]*previous[0]};
        double norm = 0;
        for (double x : normals[j]) norm += x*x;
        if (!(norm > 0)) return INFINITY;
        for (double& x : normals[j]) x /= std::sqrt(norm);
    }
    excess += std::max(0.0, longest/shortest - 100) / 100;
    for (int i = 0; i < 4; ++i)
        for (int j = i+1; j < 4; ++j) {
            double dot = 0;
            for (int k = 0; k < 3; ++k) dot += normals[i][k]*normals[j][k];
            excess += std::max(0.0, -dot - 1e-12);
        }
    return excess;
}

bool shape_ok(const std::vector<Point>& vertices, const Quad& quad) {
    return shape_excess(vertices,quad) == 0;
}

std::vector<std::vector<size_t>> collision_groups(
    const std::vector<Quad>& quads, const std::set<size_t>& affected) {
    std::set<size_t> pending = affected;
    std::vector<std::vector<size_t>> groups;
    while (!pending.empty()) {
        const size_t first = *pending.begin();
        pending.erase(pending.begin());
        std::set<uint32_t> vertices(quads[first].begin(), quads[first].end());
        std::vector<size_t> group{first};
        for (;;) {
            std::vector<size_t> found;
            for (size_t id : pending)
                for (uint32_t vertex : quads[id])
                    if (vertices.count(vertex)) { found.push_back(id); break; }
            if (found.empty()) break;
            for (size_t id : found) {
                pending.erase(id);
                group.push_back(id);
                vertices.insert(quads[id].begin(), quads[id].end());
            }
        }
        groups.push_back(std::move(group));
    }
    return groups;
}

struct Patch {
    std::vector<Point> vertices;
    std::vector<Quad> quads;
    std::vector<uint32_t> global_vertices;
    std::vector<std::vector<size_t>> incident;
    std::vector<uint32_t> movable;
};

Patch make_patch(const std::vector<Point>& vertices,
                 const std::vector<Quad>& quads,
                 const std::vector<size_t>& group, double padding) {
    std::set<uint32_t> moving;
    for (size_t qi : group) moving.insert(quads[qi].begin(), quads[qi].end());
    std::array<double,3> lo{}, hi{};
    lo.fill(std::numeric_limits<double>::infinity());
    hi.fill(-std::numeric_limits<double>::infinity());
    for (uint32_t vertex : moving)
        for (int k = 0; k < 3; ++k) {
            lo[k] = std::min(lo[k], double(vertices[vertex][k])-padding);
            hi[k] = std::max(hi[k], double(vertices[vertex][k])+padding);
        }
    std::vector<size_t> selected;
    for (size_t qi = 0; qi < quads.size(); ++qi) {
        bool overlap = true;
        for (int k = 0; k < 3 && overlap; ++k) {
            double a = std::numeric_limits<double>::infinity();
            double b = -std::numeric_limits<double>::infinity();
            for (uint32_t vertex : quads[qi]) {
                a = std::min(a, double(vertices[vertex][k]));
                b = std::max(b, double(vertices[vertex][k]));
            }
            overlap = a <= hi[k] && b >= lo[k];
        }
        if (overlap) selected.push_back(qi);
    }
    Patch patch;
    std::set<uint32_t> used;
    for (size_t qi : selected) used.insert(quads[qi].begin(), quads[qi].end());
    patch.global_vertices.assign(used.begin(), used.end());
    std::map<uint32_t,uint32_t> local;
    for (uint32_t i = 0; i < patch.global_vertices.size(); ++i) {
        local[patch.global_vertices[i]] = i;
        patch.vertices.push_back(vertices[patch.global_vertices[i]]);
    }
    patch.incident.resize(patch.vertices.size());
    for (size_t qi : selected) {
        Quad quad = quads[qi];
        for (uint32_t& vertex : quad) vertex = local.at(vertex);
        const size_t index = patch.quads.size();
        patch.quads.push_back(quad);
        for (uint32_t vertex : quad) patch.incident[vertex].push_back(index);
    }
    for (uint32_t vertex : moving) patch.movable.push_back(local.at(vertex));
    return patch;
}

double movement_limit(const std::vector<Point>& original, const Patch& patch,
                      uint32_t local, const QuadLocalRepairOptions& options) {
    const uint32_t global = patch.global_vertices[local];
    double shortest = std::numeric_limits<double>::infinity();
    for (size_t qi : patch.incident[local]) {
        const auto& quad = patch.quads[qi];
        for (int j = 0; j < 4; ++j)
            if (quad[j] == local)
                for (uint32_t neighbor : {quad[(j+1)%4],quad[(j+3)%4]})
                    shortest = std::min(shortest,distance(original[global],
                        original[patch.global_vertices[neighbor]]));
    }
    return std::min(options.maximum_vertex_move,
                    options.maximum_incident_edge_fraction*shortest);
}

std::vector<Point> candidate_positions(const Point& point, const Point& target) {
    std::vector<Point> candidates;
    const auto add = [&](const Point& candidate) {
        if (candidate == point || std::find(candidates.begin(),candidates.end(),candidate) != candidates.end())
            return;
        candidates.push_back(candidate);
    };
    for (int axis = 0; axis < 3; ++axis)
        for (float direction : {-INFINITY, INFINITY}) {
            Point candidate = point;
            candidate[axis] = std::nextafter(point[axis], direction);
            add(candidate);
        }
    for (double fraction : {0.001,0.01,0.05}) {
        Point candidate;
        for (int k = 0; k < 3; ++k)
            candidate[k] = float(double(point[k]) + fraction*(double(target[k])-point[k]));
        add(candidate);
    }
    return candidates;
}

}

QuadShapeRepairResult repair_quad_shapes(
    std::vector<Point>& vertices, const std::vector<Quad>& quads,
    const QuadLocalRepairOptions& options) {
    if (vertices.empty() || quads.empty() ||
        !(options.neighborhood_padding > 0 && options.neighborhood_padding <= 0.008) ||
        !(options.maximum_vertex_move > 0 && options.maximum_vertex_move <= options.neighborhood_padding) ||
        !(options.maximum_incident_edge_fraction > 0 &&
          options.maximum_incident_edge_fraction <= 0.1) ||
        !options.maximum_steps_per_group)
        throw std::invalid_argument("Invalid quad shape repair input");
    std::set<size_t> affected;
    for (size_t qi = 0; qi < quads.size(); ++qi)
        if (shape_excess(vertices,quads[qi]) > 0) affected.insert(qi);
    QuadShapeRepairResult result;
    result.bad_quads_before = affected.size();
    if (affected.empty()) return result;
    auto candidate_mesh = vertices;
    for (const auto& group : collision_groups(quads,affected)) {
        Patch patch = make_patch(candidate_mesh,quads,group,options.neighborhood_padding);
        const auto score = [&]() {
            size_t bad = 0;
            double excess = 0;
            for (const auto& quad : patch.quads) {
                const double value = shape_excess(patch.vertices,quad);
                if (value > 0) { ++bad; excess += value; }
            }
            return std::pair(bad,excess);
        };
        auto current = score();
        auto collision = audit_quad_collisions(patch.vertices,patch.quads);
        if (collision.duplicate_triangle_pairs || collision.diagonal_edge_conflicts)
            throw std::runtime_error("Shape patch has nonrepairable topology defects");
        for (size_t step = 0; step < options.maximum_steps_per_group && current.first; ++step) {
            uint32_t best_vertex = std::numeric_limits<uint32_t>::max();
            Point best_position{};
            auto best_score = current;
            for (uint32_t local : patch.movable) {
                const Point previous = patch.vertices[local];
                const Point initial = vertices[patch.global_vertices[local]];
                const auto positions = candidate_positions(previous,previous);
                for (size_t choice = 0; choice < std::min(size_t(6),positions.size()); ++choice) {
                    const Point& trial = positions[choice];
                    if (distance(initial,trial) > movement_limit(vertices,patch,local,options)) continue;
                    patch.vertices[local] = trial;
                    const auto trial_score = score();
                    if (trial_score < best_score) {
                        const auto trial_collision = audit_quad_collisions(patch.vertices,patch.quads);
                        if (trial_collision.cross_polygon_intersections <= collision.cross_polygon_intersections &&
                            trial_collision.degenerate_triangles <= collision.degenerate_triangles) {
                            best_score = trial_score;
                            best_vertex = local;
                            best_position = trial;
                        }
                    }
                }
                patch.vertices[local] = previous;
            }
            if (best_vertex == std::numeric_limits<uint32_t>::max()) break;
            const uint32_t global = patch.global_vertices[best_vertex];
            const Point before = candidate_mesh[global];
            candidate_mesh[global] = best_position;
            patch.vertices[best_vertex] = best_position;
            current = best_score;
            collision = audit_quad_collisions(patch.vertices,patch.quads);
            result.moves.push_back({global,before,best_position,collision.cross_polygon_intersections});
        }
        if (current.first)
            throw std::runtime_error("Protected local quad shape repair did not converge");
    }
    for (size_t qi = 0; qi < quads.size(); ++qi)
        result.bad_quads_after += shape_excess(candidate_mesh,quads[qi]) > 0;
    if (result.bad_quads_after)
        throw std::runtime_error("Quad shape repair failed full-mesh shape audit");
    vertices = std::move(candidate_mesh);
    return result;
}

QuadLocalRepairResult repair_quad_collisions(
    std::vector<Point>& vertices, const std::vector<Quad>& quads,
    const QuadSourceProjection& project_to_source,
    const QuadLocalRepairOptions& options) {
    if (vertices.empty() || quads.empty() || !project_to_source ||
        !(options.neighborhood_padding > 0 && options.neighborhood_padding <= 0.008) ||
        !(options.maximum_vertex_move > 0 && options.maximum_vertex_move <= options.neighborhood_padding) ||
        !(options.maximum_incident_edge_fraction > 0 &&
          options.maximum_incident_edge_fraction <= 0.1) ||
        !options.maximum_steps_per_group)
        throw std::invalid_argument("Invalid local quad repair input");
    QuadLocalRepairResult result;
    result.before = audit_quad_collisions(vertices,quads);
    if (result.before.clean()) { result.after = result.before; return result; }
    if (result.before.duplicate_triangle_pairs || result.before.diagonal_edge_conflicts)
        throw std::runtime_error("Local vertex repair cannot resolve duplicate or chord topology");
    auto candidate_mesh = vertices;
    const auto groups = collision_groups(quads,result.before.affected_polygons);
    for (const auto& group : groups) {
        Patch patch = make_patch(candidate_mesh,quads,group,options.neighborhood_padding);
        std::map<uint32_t,Point> targets;
        for (uint32_t local : patch.movable)
            targets[local] = project_to_source(patch.vertices[local]);
        auto current = audit_quad_collisions(patch.vertices,patch.quads);
        for (size_t step = 0; step < options.maximum_steps_per_group && !current.clean(); ++step) {
            std::vector<uint32_t> order = patch.movable;
            std::sort(order.begin(),order.end(),[&](uint32_t a, uint32_t b) {
                auto count = [&](uint32_t v) {
                    size_t n = 0;
                    for (size_t qi : patch.incident[v]) n += current.affected_polygons.count(qi);
                    return n;
                };
                const size_t na = count(a), nb = count(b);
                return na != nb ? na > nb : patch.global_vertices[a] < patch.global_vertices[b];
            });
            const auto score = [](const QuadCollisionReport& report) {
                return std::pair(report.cross_polygon_intersections,report.degenerate_triangles);
            };
            auto best_score = score(current);
            uint32_t best_vertex = std::numeric_limits<uint32_t>::max();
            Point best_position{};
            QuadCollisionReport best_report;
            for (uint32_t local : order) {
                const Point original = patch.vertices[local];
                const Point initial = vertices[patch.global_vertices[local]];
                for (const Point& trial : candidate_positions(original,targets.at(local))) {
                    if (distance(initial,trial) > movement_limit(vertices,patch,local,options)) continue;
                    patch.vertices[local] = trial;
                    bool valid = true;
                    for (size_t qi : patch.incident[local])
                        if (!shape_ok(patch.vertices,patch.quads[qi])) { valid = false; break; }
                    if (valid) {
                        auto report = audit_quad_collisions(patch.vertices,patch.quads);
                        if (score(report) < best_score) {
                            best_score = score(report);
                            best_vertex = local;
                            best_position = trial;
                            best_report = std::move(report);
                        }
                    }
                }
                patch.vertices[local] = original;
            }
            if (best_vertex == std::numeric_limits<uint32_t>::max()) break;
            const uint32_t global = patch.global_vertices[best_vertex];
            const Point before = candidate_mesh[global];
            candidate_mesh[global] = best_position;
            patch.vertices[best_vertex] = best_position;
            current = std::move(best_report);
            result.moves.push_back({global,before,best_position,current.cross_polygon_intersections});
        }
        if (!current.clean())
            throw std::runtime_error("Protected local collision repair did not converge");
    }
    result.after = audit_quad_collisions(candidate_mesh,quads);
    if (!result.after.clean())
        throw std::runtime_error("Local repair failed full-mesh collision audit");
    vertices = std::move(candidate_mesh);
    return result;
}

}
