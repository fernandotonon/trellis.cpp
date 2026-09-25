#include "quad_refinement_io.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace trellis {

void write_quad_refinement_artifacts(
    const std::string& output_path,
    const QuadRefinementResult& result,
    const std::vector<size_t>& parent_face_sizes) {
    if (result.quads.empty() || result.quads.size() != result.parents.size() ||
        result.quads.size() != result.quality.size() ||
        result.vertices.size() != result.vertex_witnesses.size())
        throw std::invalid_argument("Invalid refinement result");
    std::ofstream output(output_path), quality(output_path+".quality.tsv"),
                  witness(output_path+".vertex-witness.tsv"),
                  splits(output_path+".split-edges.tsv");
    if (!output || !quality || !witness || !splits)
        throw std::runtime_error("Cannot open refinement output");
    for (const auto& edge : result.split_edges)
        splits << edge[0] << '\t' << edge[1] << '\n';
    witness << std::setprecision(17);
    for (size_t i = 0; i < result.vertex_witnesses.size(); ++i) {
        witness << i;
        for (const auto& [source, weight] : result.vertex_witnesses[i])
            witness << '\t' << source << ' ' << weight;
        witness << '\n';
    }
    output << std::setprecision(9);
    quality << std::setprecision(17);
    for (const auto& point : result.vertices)
        output << "v " << point[0] << ' ' << point[1] << ' ' << point[2] << '\n';
    size_t bad = 0, from_triangles = 0, nonpositive = 0;
    for (size_t i = 0; i < result.quads.size(); ++i) {
        const auto& quad = result.quads[i];
        if (result.parents[i] >= parent_face_sizes.size())
            throw std::invalid_argument("Invalid refinement parent");
        output << "f";
        std::array<std::array<float, 3>, 4> points;
        for (int corner = 0; corner < 4; ++corner) {
            if (quad[corner] >= result.vertices.size())
                throw std::invalid_argument("Invalid refined vertex");
            output << ' ' << quad[corner]+1;
            points[corner] = result.vertices[quad[corner]];
        }
        output << '\n';
        quality << i << '\t' << result.parents[i] << '\t' << parent_face_sizes[result.parents[i]]
                << '\t' << result.quality[i] << '\t'
                << quad_corner_normal_span_degrees(points) << '\n';
        if (!(result.quality[i] >= .1)) {
            ++bad;
            from_triangles += parent_face_sizes[result.parents[i]] == 3;
        }
        nonpositive += !(result.quality[i] > 0);
    }
    if (!output || !quality || !witness || !splits)
        throw std::runtime_error("Cannot write refinement output");
    std::cout << "paths=" << result.parity_paths << " unchanged=" << result.unchanged
              << " opposite=" << result.opposite << " adjacent=" << result.adjacent
              << " full=" << result.full << '\n';
    std::cout << "rotated_for_source_coverage=" << result.rotated_for_source_coverage
              << " unproven_quads=" << result.unproven_quads << '\n';
    std::cout << "optimized_centers=" << result.optimized_centers << '\n';
    std::cout << "quads=" << result.quads.size() << " below_threshold=" << bad
              << " from_triangles=" << from_triangles << " nonpositive=" << nonpositive
              << " minimum=" << *std::min_element(result.quality.begin(), result.quality.end())
              << '\n';
}

}
