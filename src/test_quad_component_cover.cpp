#include "quad_components.h"

#include <array>
#include <cstdint>
#include <exception>
#include <iostream>
#include <set>
#include <vector>

int main() {
    using Point = std::array<float,3>;
    const std::vector<Point> quads_vertices{
        {0,0,0},{.001f,0,0},{.001f,.001f,0},{0,.001f,0},
        {.1f,0,0},{.101f,0,0},{.101f,.001f,0},{.1f,.001f,0}
    };
    const std::vector<std::array<uint32_t,4>> quads{{0,1,2,3},{4,5,6,7}};
    const auto labels=trellis::label_quad_components(quads_vertices.size(),quads);
    const std::vector<Point> source{quads_vertices[4],quads_vertices[5],quads_vertices[6],quads_vertices[7]};
    const std::vector<std::array<int32_t,3>> faces{{0,1,2},{0,2,3}};
    const auto cover=trellis::select_source_covering_quad_components(
        source,faces,quads_vertices,quads,labels,.008);
    if (cover.keep!=std::set<size_t>{1} || cover.maximum_leaf_upper_bound>.008 ||
        cover.maximum_reverse_leaf_upper_bound>.008) {
        std::cerr << "Source-cover selection retained the wrong component\n";
        return 1;
    }
    bool rejected=false;
    try {
        const std::vector<Point> distant{{.5f,0,0},{.501f,0,0},{.501f,.001f,0}};
        trellis::select_source_covering_quad_components(
            distant,{{0,1,2}},quads_vertices,quads,labels,.008);
    } catch (const std::exception&) { rejected=true; }
    if (!rejected) {
        std::cerr << "Unreachable source surface was accepted\n";
        return 1;
    }
    rejected=false;
    try {
        const std::vector<Point> warped{{0,0,0},{.1f,0,0},{.1f,.1f,.1f},{0,.1f,0}};
        const std::vector<std::array<uint32_t,4>> one_quad{{0,1,2,3}};
        const auto one_label=trellis::label_quad_components(warped.size(),one_quad);
        trellis::select_source_covering_quad_components(
            warped,{{0,1,2},{0,2,3}},warped,one_quad,one_label,.008);
    } catch (const std::exception&) { rejected=true; }
    if (!rejected) {
        std::cerr << "An over-budget alternate diagonal was accepted\n";
        return 1;
    }
    rejected=false;
    try {
        const std::vector<Point> protruding{
            {0,0,0},{.001f,0,0},{.001f,.001f,0},{0,.001f,0},
            {.1f,0,0},{.1f,.001f,0}
        };
        const std::vector<std::array<uint32_t,4>> two_quads{{0,1,2,3},{1,4,5,2}};
        const auto one_component=trellis::label_quad_components(protruding.size(),two_quads);
        const std::vector<Point> near_source{
            protruding[0],protruding[1],protruding[2],protruding[3]
        };
        trellis::select_source_covering_quad_components(
            near_source,{{0,1,2},{0,2,3}},protruding,two_quads,one_component,.008);
    } catch (const std::exception&) { rejected=true; }
    if (!rejected) {
        std::cerr << "A detached extension of a retained component was accepted\n";
        return 1;
    }
    std::cout << "source-cover component selection passed\n";
}
