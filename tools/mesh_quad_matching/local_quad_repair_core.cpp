#include "quad_local_repair.h"
#include "tri_bvh.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using Point = std::array<float,3>;
using Quad = std::array<uint32_t,4>;

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "usage: trellis-repair-core INPUT_QUADS.obj SOURCE.post OUTPUT_QUADS.obj\n";
        return 2;
    }
    try {
        std::ifstream mesh_input(argv[1]);
        if (!mesh_input) throw std::runtime_error("Cannot read input quad OBJ");
        std::vector<Point> vertices;
        std::vector<Quad> quads;
        std::string line, tag;
        while (std::getline(mesh_input,line)) {
            std::istringstream stream(line);
            if (!(stream >> tag)) continue;
            if (tag == "v") {
                Point point;
                if (!(stream >> point[0] >> point[1] >> point[2]))
                    throw std::runtime_error("Invalid vertex");
                for (float value : point)
                    if (!std::isfinite(value)) throw std::runtime_error("Nonfinite vertex");
                vertices.push_back(point);
            } else if (tag == "f") {
                Quad quad;
                for (auto& vertex : quad) {
                    size_t index;
                    if (!(stream >> index) || index == 0 || index > vertices.size())
                        throw std::runtime_error("Invalid quad");
                    vertex = uint32_t(index-1);
                }
                quads.push_back(quad);
            }
        }
        if (!mesh_input.eof() || vertices.empty() || quads.empty())
            throw std::runtime_error("Invalid quad OBJ");
        std::ifstream source(argv[2],std::ios::binary);
        int32_t header[4]{};
        source.read(reinterpret_cast<char*>(header),sizeof(header));
        if (!source || header[0] <= 0 || header[1] <= 0)
            throw std::runtime_error("Invalid source POST header");
        std::vector<Point> source_vertices(header[0]);
        std::vector<std::array<int32_t,3>> source_faces(header[1]);
        source.read(reinterpret_cast<char*>(source_vertices.data()),
                    source_vertices.size()*sizeof(Point));
        source.read(reinterpret_cast<char*>(source_faces.data()),
                    source_faces.size()*sizeof(source_faces[0]));
        if (!source) throw std::runtime_error("Truncated source POST");
        const auto shape = trellis::repair_quad_shapes(vertices,quads);
        const auto tree = trellis::TriBvh::build(
            reinterpret_cast<const float*>(source_vertices.data()),source_vertices.size(),
            reinterpret_cast<const int32_t*>(source_faces.data()),source_faces.size());
        const auto project = [&](const Point& point) -> Point {
            const auto hit = tree.closest(point.data());
            if (hit.face < 0) throw std::runtime_error("Missing source projection");
            return {hit.point[0],hit.point[1],hit.point[2]};
        };
        const auto repair = trellis::repair_quad_collisions(vertices,quads,project);
        std::ofstream output(argv[3]);
        if (!output) throw std::runtime_error("Cannot write output OBJ");
        output << std::setprecision(9);
        for (const auto& point : vertices)
            output << "v " << point[0] << ' ' << point[1] << ' ' << point[2] << '\n';
        for (const auto& quad : quads)
            output << "f " << quad[0]+1 << ' ' << quad[1]+1 << ' '
                   << quad[2]+1 << ' ' << quad[3]+1 << '\n';
        if (!output) throw std::runtime_error("Failed to write output OBJ");
        std::cout << "bad_shape_before=" << shape.bad_quads_before
                  << " bad_shape_after=" << shape.bad_quads_after
                  << " shape_moves=" << shape.moves.size() << '\n';
        for (const auto& move : shape.moves)
            std::cout << "shape_move vertex=" << move.vertex << " from="
                      << move.before[0] << ',' << move.before[1] << ',' << move.before[2]
                      << " to=" << move.after[0] << ',' << move.after[1] << ',' << move.after[2] << '\n';
        std::cout << "crossings_before=" << repair.before.cross_polygon_intersections
                  << " crossings_after=" << repair.after.cross_polygon_intersections
                  << " moved_vertices=" << repair.moves.size() << '\n';
        for (const auto& move : repair.moves)
            std::cout << "move vertex=" << move.vertex << " from="
                      << move.before[0] << ',' << move.before[1] << ',' << move.before[2]
                      << " to=" << move.after[0] << ',' << move.after[1] << ',' << move.after[2]
                      << " crossings=" << move.remaining_crossings << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
