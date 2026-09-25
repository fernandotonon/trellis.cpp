#include "quad_topology.h"

#include <array>
#include <stdexcept>
#include <cstdint>
#include <iostream>
#include <utility>
#include <vector>

void require(bool value) {
    if (!value) throw std::runtime_error("Quad topology test failed");
}

int main() {
    const std::vector<std::array<uint32_t,4>> cube{
        {0,1,2,3}, {5,4,7,6}, {4,0,3,7},
        {1,5,6,2}, {3,2,6,7}, {4,5,1,0}};
    const auto clean = trellis::audit_quad_topology(8,cube);
    require(clean.clean() && clean.components == 1);
    auto open = cube;
    open.pop_back();
    require(trellis::audit_quad_topology(8,open).boundary_edges == 4);
    auto reversed = cube;
    std::swap(reversed[0][1],reversed[0][3]);
    require(trellis::audit_quad_topology(8,reversed).winding_conflicts == 4);
    auto duplicate = cube;
    duplicate.push_back(cube[0]);
    require(trellis::audit_quad_topology(8,duplicate).duplicate_faces == 1);
    auto joined = cube;
    for (auto q : cube) {
        for (uint32_t& vertex : q) if (vertex) vertex += 7;
        joined.push_back(q);
    }
    const auto bowtie = trellis::audit_quad_topology(15,joined);
    require(bowtie.components == 2 && bowtie.nonmanifold_vertices == 1);
    std::cout << "quad topology audit passed\n";
}
