#include "uv_overlap_cover.h"
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 5) return 2;
    try {
        const size_t face_count = std::stoull(argv[1]);
        std::ifstream pair_file(argv[2]), required_file(argv[3]);
        if (!pair_file || !required_file) throw std::runtime_error("Cannot read overlap inputs");
        std::vector<std::array<uint32_t, 2>> overlaps;
        uint32_t a, b;
        while (pair_file >> a >> b) overlaps.push_back({a, b});
        if (!pair_file.eof()) throw std::runtime_error("Invalid overlap pair file");
        std::vector<uint32_t> required;
        while (required_file >> a) required.push_back(a);
        if (!required_file.eof()) throw std::runtime_error("Invalid required face file");
        const auto cover = trellis::greedy_uv_overlap_cover(face_count, overlaps, required);
        std::ofstream output(argv[4]);
        for (auto face : cover) output << face << '\n';
        if (!output) throw std::runtime_error("Cannot write overlap cover");
        std::cout << "overlap_pairs=" << overlaps.size() << " selected_faces=" << cover.size() << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
