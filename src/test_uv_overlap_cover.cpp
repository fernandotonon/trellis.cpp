#include "uv_overlap_cover.h"
#include <array>
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

int main() {
    using Pair = std::array<uint32_t, 2>;
    const std::vector<Pair> overlaps = {{0, 1}, {0, 2}, {0, 3}, {4, 5}, {5, 6}};
    const auto cover = trellis::greedy_uv_overlap_cover(7, overlaps);
    assert(cover == std::vector<uint32_t>({0, 5}));
    for (auto [a, b] : overlaps)
        assert(std::binary_search(cover.begin(), cover.end(), a) ||
               std::binary_search(cover.begin(), cover.end(), b));
    const auto required = trellis::greedy_uv_overlap_cover(7, overlaps, {1, 2, 3, 4});
    assert(required == std::vector<uint32_t>({1, 2, 3, 4, 5}));
    bool rejected = false;
    try { trellis::greedy_uv_overlap_cover(7, {{1, 7}}); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
    std::cout << "PASS greedy UV overlap cover\n";
}
