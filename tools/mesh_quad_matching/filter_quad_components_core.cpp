#include "quad_components.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: trellis-filter-quads INPUT.obj OUTPUT.obj COMPONENT_ID...\n"
                  << "   or: trellis-filter-quads INPUT.obj OUTPUT.obj --source-cover SOURCE.post LIMIT [MAX_DEPTH]\n";
        return 2;
    }
    try {
        const bool source_cover = std::string(argv[3]) == "--source-cover";
        if (source_cover && (argc < 6 || argc > 7))
            throw std::runtime_error("Expected SOURCE.post LIMIT [MAX_DEPTH]");
        std::set<size_t> keep;
        if (!source_cover) for (int i = 3; i < argc; ++i) {
            std::string token(argv[i]);
            size_t end = 0;
            const size_t id = std::stoull(token,&end);
            if (end != token.size()) throw std::runtime_error("Invalid component ID");
            keep.insert(id);
        }
        std::ifstream input(argv[1]);
        if (!input) throw std::runtime_error("Cannot read quad OBJ");
        std::vector<std::array<float,3>> vertices;
        std::vector<std::array<uint32_t,4>> quads;
        std::string line, tag;
        while (std::getline(input,line)) {
            std::istringstream stream(line);
            if (!(stream >> tag)) continue;
            if (tag == "v") {
                std::array<float,3> point;
                if (!(stream >> point[0] >> point[1] >> point[2]))
                    throw std::runtime_error("Invalid vertex");
                vertices.push_back(point);
            } else if (tag == "f") {
                std::array<uint32_t,4> quad;
                for (auto& vertex : quad) {
                    size_t index;
                    if (!(stream >> index) || index == 0 || index > vertices.size())
                        throw std::runtime_error("Invalid quad");
                    vertex = uint32_t(index-1);
                }
                quads.push_back(quad);
            }
        }
        if (!input.eof()) throw std::runtime_error("Failed to read quad OBJ");
        const auto labels = trellis::label_quad_components(vertices.size(),quads);
        if (source_cover) {
            std::ifstream source(argv[4],std::ios::binary);
            int32_t header[4]{};
            source.read(reinterpret_cast<char*>(header),sizeof(header));
            if (!source || header[0] <= 0 || header[1] <= 0)
                throw std::runtime_error("Invalid source POST header");
            std::vector<std::array<float,3>> source_vertices(header[0]);
            std::vector<std::array<int32_t,3>> source_faces(header[1]);
            source.read(reinterpret_cast<char*>(source_vertices.data()),source_vertices.size()*sizeof(source_vertices[0]));
            source.read(reinterpret_cast<char*>(source_faces.data()),source_faces.size()*sizeof(source_faces[0]));
            if (!source) throw std::runtime_error("Truncated source POST geometry");
            const double limit = std::stod(argv[5]);
            const int depth = argc == 7 ? std::stoi(argv[6]) : 12;
            const auto cover = trellis::select_source_covering_quad_components(
                source_vertices,source_faces,vertices,quads,labels,limit,depth);
            keep = cover.keep;
            std::cout << "source_cover_rounds=" << cover.rounds
                      << " maximum_leaf_upper_bound=" << std::setprecision(12)
                      << cover.maximum_leaf_upper_bound
                      << " reverse_upper_bound=" << cover.maximum_reverse_leaf_upper_bound
                      << " distance_queries=" << cover.distance_queries
                      << " reverse_queries=" << cover.reverse_distance_queries
                      << " selected_components=";
            for (size_t id : keep) std::cout << id << ',';
            std::cout << '\n';
        }
        const auto result = trellis::filter_quad_components(vertices,quads,labels,keep);
        std::ofstream output(argv[2]);
        if (!output) throw std::runtime_error("Cannot write filtered OBJ");
        output << std::setprecision(9);
        for (const auto& p : result.vertices)
            output << "v " << p[0] << ' ' << p[1] << ' ' << p[2] << '\n';
        for (const auto& q : result.quads)
            output << "f " << q[0]+1 << ' ' << q[1]+1 << ' '
                   << q[2]+1 << ' ' << q[3]+1 << '\n';
        if (!output) throw std::runtime_error("Failed to write filtered OBJ");
        std::cout << "components=" << labels.face_counts.size()
                  << " selected_quads=" << result.quads.size()
                  << " selected_vertices=" << result.vertices.size() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
