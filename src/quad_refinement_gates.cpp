#include "quad_refinement.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>

namespace trellis {

QuadRefinementGateReport audit_quad_refinement(
    const std::vector<std::vector<uint32_t>>& source_faces,
    const QuadRefinementResult& result) {
    if (source_faces.empty() || result.quads.size() != result.parents.size() ||
        result.quads.size() != result.quality.size() ||
        result.vertices.size() != result.vertex_witnesses.size())
        throw std::invalid_argument("Invalid refinement audit input");
    QuadRefinementGateReport report;
    std::vector<std::array<double, 2>> areas(source_faces.size());
    const std::array<std::array<int, 3>, 2> fan = {{{0,1,2},{0,2,3}}};
    for (size_t qi = 0; qi < result.quads.size(); ++qi) {
        const size_t parent = result.parents[qi];
        if (parent >= source_faces.size()) throw std::invalid_argument("Invalid source parent");
        const auto& reference = source_faces[parent];
        if (reference.size() != 3 && reference.size() != 4)
            throw std::invalid_argument("Invalid source polygon");
        const auto& quad = result.quads[qi];
        std::array<std::array<float,3>,4> points;
        for (int j = 0; j < 4; ++j) {
            if (quad[j] >= result.vertices.size()) throw std::invalid_argument("Invalid quad vertex");
            points[j] = result.vertices[quad[j]];
        }
        if (!std::isfinite(result.quality[qi]) || result.quality[qi] < .1)
            report.bad_quality_parents.insert(parent);
        double shortest = 1e300, longest = 0, minimum_angle = 180;
        for (int j = 0; j < 4; ++j) {
            std::array<double,3> next{}, previous{};
            double next_squared = 0, previous_squared = 0, dot = 0;
            for (int axis = 0; axis < 3; ++axis) {
                next[axis] = double(points[(j+1)%4][axis])-points[j][axis];
                previous[axis] = double(points[(j+3)%4][axis])-points[j][axis];
                next_squared += next[axis]*next[axis];
                previous_squared += previous[axis]*previous[axis];
                dot += next[axis]*previous[axis];
            }
            const double length = std::sqrt(next_squared);
            shortest = std::min(shortest, length);
            longest = std::max(longest, length);
            if (!(next_squared > 0 && previous_squared > 0)) {
                minimum_angle = 0;
                continue;
            }
            const double cosine = std::clamp(dot/std::sqrt(next_squared*previous_squared), -1.0, 1.0);
            minimum_angle = std::min(minimum_angle,
                std::acos(cosine)*180/3.141592653589793);
        }
        const double ratio = shortest > 0 ? longest/shortest : INFINITY;
        if (quad_corner_normal_span_degrees(points) > 90 || ratio > 100 || minimum_angle < 1)
            report.bad_shape_parents.insert(parent);

        for (const auto& triangle : fan) {
            std::set<size_t> support;
            for (int corner : triangle)
                for (const auto& [source, weight] : result.vertex_witnesses[quad[corner]]) {
                    if (source >= result.vertices.size() || !std::isfinite(weight))
                        throw std::invalid_argument("Invalid vertex witness");
                    support.insert(source);
                }
            bool found = false;
            for (size_t k = 1; k+1 < reference.size(); ++k) {
                const std::set<size_t> ref = {reference[0],reference[k],reference[k+1]};
                if (!std::includes(ref.begin(),ref.end(),support.begin(),support.end())) continue;
                double matrix[3][3]{};
                const size_t columns[3] = {reference[0],reference[k],reference[k+1]};
                for (int row = 0; row < 3; ++row)
                    for (const auto& [source,weight] : result.vertex_witnesses[quad[triangle[row]]])
                        for (int col = 0; col < 3; ++col)
                            if (source == columns[col]) matrix[row][col] += weight;
                const double det =
                    matrix[0][0]*(matrix[1][1]*matrix[2][2]-matrix[1][2]*matrix[2][1])-
                    matrix[0][1]*(matrix[1][0]*matrix[2][2]-matrix[1][2]*matrix[2][0])+
                    matrix[0][2]*(matrix[1][0]*matrix[2][1]-matrix[1][1]*matrix[2][0]);
                areas[parent][k-1] += det;
                if (!(det > 0)) ++report.nonpositive_witness_triangles;
                found = true;
                break;
            }
            if (!found) report.unproven_parents.insert(parent);
        }
    }
    for (size_t parent = 0; parent < source_faces.size(); ++parent)
        for (size_t k = 0; k+2 < source_faces[parent].size(); ++k)
            if (std::abs(areas[parent][k]-1) > 1e-10)
                report.unproven_parents.insert(parent);
    return report;
}

QualifiedQuadRefinementResult refine_quads_until_qualified(
    const std::vector<std::array<float, 3>>& vertices,
    const std::vector<std::vector<uint32_t>>& faces,
    QuadRefinementOptions options,
    size_t maximum_rounds,
    bool permit_shape_repair) {
    if (!maximum_rounds) throw std::invalid_argument("Zero refinement rounds");
    QualifiedQuadRefinementResult qualified;
    for (size_t round = 0; round < maximum_rounds; ++round) {
        auto mesh = refine_quads(vertices,faces,options);
        const auto gate = audit_quad_refinement(faces,mesh);
        std::set<size_t> failed = gate.unproven_parents;
        failed.insert(gate.bad_quality_parents.begin(),gate.bad_quality_parents.end());
        failed.insert(gate.bad_shape_parents.begin(),gate.bad_shape_parents.end());
        size_t added = 0;
        for (size_t parent : failed) added += options.forced_faces.insert(parent).second;
        qualified.rounds.push_back({options.forced_faces.size()-added, added,
            gate.bad_quality_parents.size(), gate.bad_shape_parents.size(),
            gate.unproven_parents.size(), mesh.quads.size()});
        if (!added) {
            const bool shape_only = gate.unproven_parents.empty() &&
                                    gate.bad_quality_parents.empty() &&
                                    !gate.bad_shape_parents.empty() &&
                                    !gate.nonpositive_witness_triangles;
            if ((!failed.empty() || gate.nonpositive_witness_triangles) &&
                !(permit_shape_repair && shape_only))
                throw std::runtime_error("Refinement stopped with unresolved gates");
            qualified.gates_passed = failed.empty() && !gate.nonpositive_witness_triangles;
            qualified.mesh = std::move(mesh);
            return qualified;
        }
        options.use_split_seed = true;
        options.split_seed = std::move(mesh.split_edges);
    }
    throw std::runtime_error("Refinement did not converge");
}

}
