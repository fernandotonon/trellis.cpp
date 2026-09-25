#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace trellis {

std::vector<uint32_t> greedy_uv_overlap_cover(
    size_t face_count, const std::vector<std::array<uint32_t, 2>>& overlaps,
    const std::vector<uint32_t>& required_faces = {});

}
