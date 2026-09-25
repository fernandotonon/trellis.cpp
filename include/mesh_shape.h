#pragma once
#include <cstdint>
#include <vector>

namespace trellis {

struct ShapeRepairStats {
    int flips = 0;
    int collision_rejections = 0;
    int swept_rejections = 0;
};

struct SmoothingStats {
    int moved = 0;
    int collision_rejections = 0;
    float maximum_displacement = 0;
};

ShapeRepairStats flip_skinny_triangles(const std::vector<float>& vertices,
                                       std::vector<int32_t>& faces,
                                       float minimum_angle_degrees = 1.0f,
                                       float maximum_diagonal_gap = 0.002f,
                                       int rings = 2, int passes = 5);

SmoothingStats smooth_skinny_triangles(std::vector<float>& vertices,
                                      const std::vector<int32_t>& faces,
                                      float minimum_angle_degrees = 1.0f,
                                      float maximum_displacement = 0.001f,
                                      int passes = 3);

}
