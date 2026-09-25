#pragma once
#include "uv_bake.h"
#include "tri_bvh.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <unordered_map>

namespace trellis::detail {
struct VoxSampler {
    std::unordered_map<uint64_t, int> map;
    const std::vector<float>* feats;
    int res;
    const TriBvh* snap;
    static uint64_t key(int x, int y, int z) {
        return ((uint64_t)(uint32_t)x << 40) | ((uint64_t)(uint32_t)y << 20) | (uint32_t)z;
    }
    explicit VoxSampler(const VoxelPbr& v) : feats(v.feats), res(v.res), snap(v.snap) {
        map.reserve(v.coords->size() * 2);
        for (size_t i = 0; i < v.coords->size(); ++i) {
            const auto& c = (*v.coords)[i];
            map[key(c[0], c[1], c[2])] = (int)i;
        }
    }
    bool trilinear(const float p[3], float out[6]) const {
        float w[3]; int b[3];
        for (int a = 0; a < 3; ++a) {
            const float gf = (p[a] + 0.5f) * res - 0.5f;
            b[a] = (int)std::floor(gf);
            w[a] = gf - b[a];
        }
        float acc[6] = {0,0,0,0,0,0}, wsum = 0.f;
        for (int dz = 0; dz < 2; ++dz) for (int dy = 0; dy < 2; ++dy) for (int dx = 0; dx < 2; ++dx) {
            const float ww = (dx ? w[0] : 1-w[0]) * (dy ? w[1] : 1-w[1]) * (dz ? w[2] : 1-w[2]);
            if (ww <= 0.f) continue;
            auto it = map.find(key(b[0]+dx, b[1]+dy, b[2]+dz));
            if (it == map.end()) continue;
            const float* f = &(*feats)[(size_t)it->second * 6];
            for (int k = 0; k < 6; ++k) acc[k] += ww * f[k];
            wsum += ww;
        }
        if (wsum <= 1e-6f) return false;
        for (int k = 0; k < 6; ++k) out[k] = acc[k] / wsum;
        return true;
    }
    bool sample(const float p[3], float out[6]) const {
        if (trilinear(p, out)) return true;
        if (snap) {
            const TriBvh::Hit h = snap->closest(p, 8.0f / res);
            if (h.face >= 0 && trilinear(h.point, out)) return true;
        }
        int b[3];
        float fraction[3];
        for (int a = 0; a < 3; ++a) {
            const float grid = (p[a] + 0.5f) * res - 0.5f;
            b[a] = (int)std::floor(grid);
            fraction[a] = grid - b[a];
        }
        for (int r = 1; r <= 3; ++r) {
            float racc[6] = {0,0,0,0,0,0}; int hits = 0;
            for (int dz = -r; dz <= r+1; ++dz) for (int dy = -r; dy <= r+1; ++dy) for (int dx = -r; dx <= r+1; ++dx) {
                const float distance = std::max({std::abs(dx-fraction[0]),
                    std::abs(dy-fraction[1]), std::abs(dz-fraction[2])});
                if (distance > r+0.5f || (r>1 && distance <= r-0.5f)) continue;
                auto it = map.find(key(b[0]+dx, b[1]+dy, b[2]+dz));
                if (it == map.end()) continue;
                const float* f = &(*feats)[(size_t)it->second * 6];
                for (int k = 0; k < 6; ++k) racc[k] += f[k];
                ++hits;
            }
            if (hits) {
                for (int k = 0; k < 6; ++k) out[k] = racc[k] / hits;
                return true;
            }
        }
        return false;
    }
};
}
