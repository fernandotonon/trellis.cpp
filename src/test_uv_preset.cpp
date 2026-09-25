#include "uv_bake.h"
#include "tri_bvh.h"
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>

int main() {
    using namespace trellis;
    BakedMesh mesh;
    mesh.T=128;
    mesh.verts={-.004f,-.004f,0, .004f,-.004f,0, .004f,.004f,0, -.004f,.004f,0};
    mesh.uv={.25f,.25f, .75f,.25f, .75f,.75f, .25f,.75f};
    mesh.faces={0,1,2, 0,2,3};
    std::vector<std::array<int,3>> coords;
    std::vector<float> feats;
    for (int z=30;z<=33;++z) for (int y=30;y<=33;++y) for (int x=30;x<=33;++x) {
        coords.push_back({x,y,z});
        feats.insert(feats.end(),{.2f,.4f,.6f,.7f,.3f,1.f});
    }
    VoxelPbr vox{&coords,&feats,64,nullptr};
    const auto stats=bake_preset_uv(mesh,vox);
    if (stats.covered_texels<3000 || stats.missing_voxel_texels ||
        mesh.base.size()!=128*128*4 || mesh.mr.size()!=mesh.base.size())
        throw std::runtime_error("Preset UV coverage failed");
    const size_t t=4*(64*128+64);
    const std::array<int,4> base{51,102,153,255},mr{0,76,178,255};
    for (int k=0;k<4;++k)
        if (std::abs(int(mesh.base[t+k])-base[k])>1 || std::abs(int(mesh.mr[t+k])-mr[k])>1)
            throw std::runtime_error("Preset UV PBR value failed");
    mesh.uv[0]=2.f;
    bool rejected=false;
    try { bake_preset_uv(mesh,vox); } catch (const std::invalid_argument&) { rejected=true; }
    if (!rejected) throw std::runtime_error("Invalid preset UV accepted");
    mesh.uv[0]=.25f;
    const float source_positions[]={-.004f,-.004f,0, .004f,-.004f,0, .004f,.004f,0, -.004f,.004f,0};
    const int32_t source_faces[]={0,1,2, 0,2,3};
    auto source=TriBvh::build(source_positions,4,source_faces,2);
    for (size_t i=2;i<mesh.verts.size();i+=3) mesh.verts[i]=.02f;
    vox.snap=&source;
    const auto projected=bake_preset_uv(mesh,vox);
    if (!projected.projected_samples || !projected.projected_samples_above_0_008 ||
        std::abs(projected.maximum_source_distance-.02)>1e-5)
        throw std::runtime_error("Preset UV source projection failed");
    std::cout << "PASS preset UV PBR raster, coverage, rejection and source projection\n";
}
