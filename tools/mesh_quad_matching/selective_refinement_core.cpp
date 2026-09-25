#include "quad_refinement.h"
#include "quad_refinement_io.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

void read_ids(const char* path, std::set<size_t>& ids) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Cannot open index list");
    size_t id;
    while (input >> id) ids.insert(id);
    if (!input.eof()) throw std::runtime_error("Invalid index list");
}

}

int main(int argc, char** argv) {
    if (argc < 3 || argc > 7) return 2;
    try {
        if (std::getenv("TRELLIS_PROBE_CENTER_SICN")) return 2;
        trellis::QuadRefinementOptions options;
        options.surface = argc >= 5;
        if (options.surface && std::string(argv[4]) != "surface") return 2;
        if (const char* value = std::getenv("TRELLIS_PROBE_CENTER_NORMAL_SPAN"))
            options.center_normal_span_degrees = std::stod(value);
        if (argc >= 4) read_ids(argv[3], options.forced_faces);
        if (argc >= 6) read_ids(argv[5], options.frozen_faces);
        if (argc == 7) {
            options.use_split_seed = true;
            std::ifstream input(argv[6]);
            if (!input) return 2;
            size_t a, b;
            while (input >> a >> b) options.split_seed.insert({std::min(a,b), std::max(a,b)});
            if (!input.eof()) return 2;
        }
        std::ifstream input(argv[1]);
        if (!input) return 2;
        std::vector<std::array<float, 3>> vertices;
        std::vector<std::vector<uint32_t>> faces;
        std::string line;
        while (std::getline(input, line)) {
            std::istringstream stream(line);
            std::string tag;
            stream >> tag;
            if (tag == "v") {
                std::array<float, 3> point;
                if (!(stream >> point[0] >> point[1] >> point[2])) return 2;
                vertices.push_back(point);
            } else if (tag == "f") {
                std::vector<uint32_t> face;
                size_t index;
                while (stream >> index) {
                    if (!index || index > vertices.size() || index > UINT32_MAX) return 2;
                    face.push_back(uint32_t(index-1));
                }
                if (!stream.eof() || (face.size() != 3 && face.size() != 4)) return 2;
                faces.push_back(std::move(face));
            }
        }
        if (!input.eof()) return 2;
        trellis::QuadRefinementResult result;
        if (std::getenv("TRELLIS_PROBE_AUDIT_ONLY")) {
            result=trellis::refine_quads(vertices,faces,options);
            const auto gate=trellis::audit_quad_refinement(faces,result);
            std::set<size_t> forced=gate.unproven_parents;
            forced.insert(gate.bad_quality_parents.begin(),gate.bad_quality_parents.end());
            forced.insert(gate.bad_shape_parents.begin(),gate.bad_shape_parents.end());
            std::ofstream next(std::string(argv[2])+".next-forced.txt");
            if (!next) return 2;
            for (size_t face : forced) next << face << '\n';
            std::cout << "AUDIT quality=" << gate.bad_quality_parents.size()
                      << " shape=" << gate.bad_shape_parents.size()
                      << " unproven=" << gate.unproven_parents.size()
                      << " nonpositive=" << gate.nonpositive_witness_triangles
                      << " next_forced=" << forced.size() << '\n';
        } else if (std::getenv("TRELLIS_PROBE_QUALIFY")) {
            auto qualified = trellis::refine_quads_until_qualified(vertices,faces,options);
            for (size_t i=0; i<qualified.rounds.size(); ++i) {
                const auto& round=qualified.rounds[i];
                std::cout << "QUALIFY round=" << i << " forced=" << round.forced_parents
                          << " new=" << round.new_forced_parents
                          << " quality=" << round.bad_quality_parents
                          << " shape=" << round.bad_shape_parents
                          << " unproven=" << round.unproven_parents
                          << " quads=" << round.quad_faces << '\n';
            }
            result=std::move(qualified.mesh);
        } else result=trellis::refine_quads(vertices, faces, options);
        std::vector<size_t> face_sizes;
        face_sizes.reserve(faces.size());
        for (const auto& face : faces) face_sizes.push_back(face.size());
        trellis::write_quad_refinement_artifacts(argv[2], result, face_sizes);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
