#include "voxel_pbr_sampler.h"
#include <array>
#include <iostream>
#include <stdexcept>
using namespace trellis;
void check(bool condition) { if (!condition) throw std::runtime_error("Voxel sampler regression"); }
std::array<float,3> world(std::array<float,3> p) {
    for (auto &x:p) x=(x+.5f)/64-.5f;
    return p;
}
int main() {
    for (int axis=0;axis<3;++axis) for (float fraction:{0.f,.25f,.5f,.75f}) {
        std::array<float,3> grid{24.25f,24.25f,24.25f};grid[axis]=24+fraction;
        for (int offset:{-4,-3,3,4,5}) {
            std::array<int,3> cell{24,24,24};cell[axis]+=offset;
            bool expected=std::abs(float(offset)-fraction)<=3.5f;
            for (int reflected=0;reflected<2;++reflected) {
                auto g=grid;auto c=cell;
                if(reflected) {g[axis]=63-g[axis];c[axis]=63-c[axis];}
                std::vector<std::array<int,3>> coords{c};std::vector<float> feats{.1f,.2f,.3f,.4f,.5f,1.f};
                VoxelPbr vox{&coords,&feats,64,nullptr};detail::VoxSampler sampler(vox);
                float out[6];auto p=world(g);check(sampler.sample(p.data(),out)==expected);
                if(expected)for(int k=0;k<6;++k)check(std::abs(out[k]-feats[k])<1e-6);
            }
        }
    }
    std::vector<std::array<int,3>> coords;std::vector<float> feats;
    for(int z=0;z<2;++z)for(int y=0;y<2;++y)for(int x=0;x<2;++x) {
        coords.push_back({24+x,24+y,24+z});for(int k=0;k<6;++k)feats.push_back(float(x));
    }
    VoxelPbr vox{&coords,&feats,64,nullptr};detail::VoxSampler sampler(vox);float out[6];auto p=world({24.25f,24.5f,24.75f});
    check(sampler.sample(p.data(),out));for(float x:out)check(std::abs(x-.25f)<1e-6);
    coords.clear();feats.clear();detail::VoxSampler empty(vox);check(!empty.sample(p.data(),out));
    std::cout<<"PASS 120 reflected/axis/bounded-neighborhood cases, trilinear oracle, empty volume\n";
}
