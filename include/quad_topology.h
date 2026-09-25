#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace trellis {

struct QuadTopologyAudit {
    size_t vertices = 0;
    size_t quads = 0;
    size_t components = 0;
    size_t boundary_edges = 0;
    size_t nonmanifold_edges = 0;
    size_t nonmanifold_vertices = 0;
    size_t winding_conflicts = 0;
    size_t duplicate_faces = 0;

    bool clean() const {
        return !boundary_edges && !nonmanifold_edges && !nonmanifold_vertices &&
               !winding_conflicts && !duplicate_faces;
    }
};

QuadTopologyAudit audit_quad_topology(
    size_t vertex_count, const std::vector<std::array<uint32_t, 4>>& quads);

}
