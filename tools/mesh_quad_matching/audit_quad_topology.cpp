#include "quad_topology.h"

#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: trellis-audit-quad-topology QUADS.obj REPORT.json\n";
        return 2;
    }
    try {
        std::ifstream input(argv[1]);
        if (!input) throw std::runtime_error("Cannot read quad OBJ");
        size_t vertices = 0;
        std::vector<std::array<uint32_t,4>> quads;
        std::string line,tag;
        while (std::getline(input,line)) {
            std::istringstream row(line);
            if (!(row >> tag)) continue;
            if (tag == "v") {
                float p[3];
                if (!(row >> p[0] >> p[1] >> p[2]))
                    throw std::runtime_error("Invalid OBJ vertex");
                ++vertices;
            } else if (tag == "f") {
                std::array<uint32_t,4> q;
                for (auto& value : q) {
                    uint64_t index;
                    if (!(row >> index) || !index || index > vertices)
                        throw std::runtime_error("Invalid OBJ quad index");
                    value = uint32_t(index-1);
                }
                std::string extra;
                if (row >> extra) throw std::runtime_error("Non-quad OBJ face");
                quads.push_back(q);
            }
        }
        if (!input.eof()) throw std::runtime_error("Failed reading quad OBJ");
        const auto audit = trellis::audit_quad_topology(vertices,quads);
        std::ofstream report(argv[2]);
        if (!report) throw std::runtime_error("Cannot write topology report");
        report << "{\"vertices\":" << audit.vertices
               << ",\"quads\":" << audit.quads
               << ",\"components\":" << audit.components
               << ",\"boundary_edges\":" << audit.boundary_edges
               << ",\"nonmanifold_edges\":" << audit.nonmanifold_edges
               << ",\"nonmanifold_vertices\":" << audit.nonmanifold_vertices
               << ",\"winding_conflicts\":" << audit.winding_conflicts
               << ",\"duplicate_faces\":" << audit.duplicate_faces
               << ",\"accepted\":" << (audit.clean()?"true":"false") << "}\n";
        if (!report) throw std::runtime_error("Cannot finish topology report");
        std::cout << "quads=" << audit.quads << " components=" << audit.components
                  << " topology_accepted=" << audit.clean() << '\n';
        return audit.clean()?0:1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
