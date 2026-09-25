#include "quad_uv_repair.h"
#include "tri_bvh.h"
#include "uv_bake.h"
#include "mesh_glb.h"
#include "stb_image_write.h"
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <locale>
#include <stdexcept>
#include <string>
#include <vector>

static void write_textured_quads(const std::string& glb_path, const UvQuads& quads,
    const trellis::BakedMesh& bm, const std::vector<UvPoint3>& positions,
    const std::vector<std::array<uint32_t,3>>& faces,
    const std::vector<std::array<uint32_t,2>>& mapping) {
    const std::filesystem::path prefix=std::filesystem::path(glb_path).replace_extension();
    const std::string stem=prefix.filename().string();
    const auto obj=prefix.string()+".quads.obj";
    const auto mtl=prefix.string()+".quads.mtl";
    std::vector<float> rotated(bm.verts.size());
    for (size_t i=0;i<bm.verts.size()/3;++i) {
        rotated[3*i]=bm.verts[3*i];rotated[3*i+1]=bm.verts[3*i+2];
        rotated[3*i+2]=-bm.verts[3*i+1];
    }
    const auto normals=trellis::welded_normals_gltf(rotated.data(),int64_t(rotated.size()/3),
                                                    bm.faces.data(),int64_t(bm.faces.size()/3));
    std::ofstream output(obj);output.imbue(std::locale::classic());output<<std::setprecision(9);
    output<<"mtllib "<<stem<<".quads.mtl\nusemtl retopo\n";
    for (auto p:quads.p) output<<"v "<<p[0]<<' '<<p[2]<<' '<<-p[1]<<'\n';
    for (size_t i=0;i<quads.q.size();++i) {
        const auto q=quads.q[i];const auto pair=mapping[i];
        std::array<uint32_t,4> corner;
        for (int j=0;j<4;++j)
            corner[j]=uv_vertex_at(positions,faces[pair[j==3?1:0]],quads.p[q[j]]);
        for (int j:{0,2}) {
            const auto other=uv_vertex_at(positions,faces[pair[1]],quads.p[q[j]]);
            if (bm.uv[2*corner[j]]!=bm.uv[2*other] || bm.uv[2*corner[j]+1]!=bm.uv[2*other+1])
                throw std::runtime_error("Internal quad UV seam");
            for (int k=0;k<3;++k)
                if (normals[3*corner[j]+k]!=normals[3*other+k])
                    throw std::runtime_error("Internal quad normal seam");
        }
        bool positive=true,negative=true;
        for (int j=0;j<4;++j) {
            const auto a=corner[j],b=corner[(j+1)%4],c=corner[(j+2)%4];
            const double ux=double(bm.uv[2*b])-bm.uv[2*a],uy=double(bm.uv[2*b+1])-bm.uv[2*a+1];
            const double vx=double(bm.uv[2*c])-bm.uv[2*b],vy=double(bm.uv[2*c+1])-bm.uv[2*b+1];
            const double turn=ux*vy-uy*vx;
            positive&=turn>0;negative&=turn<0;
        }
        if (!positive && !negative) throw std::runtime_error("Nonconvex quad UV");
        for (auto v:corner) output<<"vt "<<bm.uv[2*v]<<' '<<bm.uv[2*v+1]<<'\n';
        for (auto v:corner) output<<"vn "<<normals[3*v]<<' '<<normals[3*v+1]<<' '<<normals[3*v+2]<<'\n';
        output<<"f";
        for (int j=0;j<4;++j) output<<' '<<q[j]+1<<'/'<<4*i+j+1<<'/'<<4*i+j+1;
        output<<'\n';
    }
    output.close();if (!output) throw std::runtime_error("Cannot write textured quad OBJ");
    const size_t pixels=size_t(bm.T)*bm.T;
    std::vector<uint8_t> base(bm.base.size()),rough(pixels),metal(pixels);
    for (int y=0;y<bm.T;++y) for (int x=0;x<bm.T;++x) {
        const size_t src=size_t(y)*bm.T+x,dst=size_t(bm.T-1-y)*bm.T+x;
        for (int c=0;c<4;++c) base[4*dst+c]=bm.base[4*src+c];
        rough[dst]=bm.mr[4*src+1];metal[dst]=bm.mr[4*src+2];
    }
    if (!stbi_write_png((prefix.string()+"_base.png").c_str(),bm.T,bm.T,4,base.data(),bm.T*4) ||
        !stbi_write_png((prefix.string()+"_roughness.png").c_str(),bm.T,bm.T,1,rough.data(),bm.T) ||
        !stbi_write_png((prefix.string()+"_metallic.png").c_str(),bm.T,bm.T,1,metal.data(),bm.T))
        throw std::runtime_error("Cannot write quad textures");
    std::ofstream material(mtl);
    material<<"newmtl retopo\nKd 1 1 1\nPr 1\nPm 1\nmap_Kd "<<stem<<"_base.png\n"
            <<"map_Pr "<<stem<<"_roughness.png\nmap_Pm "<<stem<<"_metallic.png\n";
    material.close();if (!material) throw std::runtime_error("Cannot write quad MTL");
}

int main(int argc, char** argv) {
    if (argc != 6 && argc != 7) {
        std::cerr << "usage: bake-current-quad-probe TEXTURED.post QUADS.obj OUTPUT.glb ATLAS chart|box|xatlas|packed [UV.bin]\n";
        return 2;
    }
    try {
        const std::filesystem::path post_path(argv[1]);
        std::ifstream input(post_path, std::ios::binary);
        std::array<int32_t, 4> h{};
        input.read(reinterpret_cast<char*>(h.data()), sizeof(h));
        if (!input || h[0] <= 0 || h[1] <= 0 || h[2] <= 0 || h[3] <= 0 ||
            std::filesystem::file_size(post_path) != 16 + uint64_t(h[0])*12 + uint64_t(h[1])*12 + uint64_t(h[2])*36)
            throw std::runtime_error("Invalid textured POST dump");
        std::vector<float> source(size_t(h[0])*3), pbr(size_t(h[2])*6);
        std::vector<int32_t> source_faces(size_t(h[1])*3);
        std::vector<std::array<int,3>> coords(h[2]);
        input.read(reinterpret_cast<char*>(source.data()), source.size()*4);
        input.read(reinterpret_cast<char*>(source_faces.data()), source_faces.size()*4);
        input.read(reinterpret_cast<char*>(coords.data()), coords.size()*12);
        input.read(reinterpret_cast<char*>(pbr.data()), pbr.size()*4);
        if (!input) throw std::runtime_error("Truncated POST dump");
        for (float x : source) if (!std::isfinite(x)) throw std::runtime_error("Nonfinite source vertex");
        for (int32_t i : source_faces) if (i < 0 || i >= h[0]) throw std::runtime_error("Invalid source face");
        for (float x : pbr) if (!std::isfinite(x) || x < 0 || x > 1) throw std::runtime_error("Invalid PBR value");
        for (const auto& c : coords) for (int i : c) if (i < 0 || i >= h[3]) throw std::runtime_error("Invalid PBR coordinate");
        const auto quad = read_uv_quads(argv[2]);
        std::vector<float> vertices;
        vertices.reserve(quad.p.size()*3);
        for (const auto& p : quad.p) vertices.insert(vertices.end(), p.begin(), p.end());
        std::vector<int32_t> triangles;
        triangles.reserve(quad.q.size()*6);
        for (const auto& q : quad.q)
            for (int k : {0,1}) {
                const int corners[2][3] = {{0,1,2},{0,2,3}};
                for (int j : corners[k]) triangles.push_back(int32_t(q[j]));
            }
        auto bvh = trellis::TriBvh::build(source.data(), h[0], source_faces.data(), h[1]);
        trellis::VoxelPbr vox{&coords, &pbr, h[3], &bvh};
        const int atlas = std::stoi(argv[4]);
        if (atlas < 128 || atlas > 4096) throw std::runtime_error("Invalid atlas size");
        const std::string mode(argv[5]);
        trellis::BakedMesh bm;
        trellis::PresetUvBakeStats preset_stats;
        if (mode == "packed" && argc==7) {
            std::ifstream packed(argv[6],std::ios::binary);uint32_t counts[2]{};
            packed.read(reinterpret_cast<char*>(counts),8);
            if (!packed || !counts[0] || !counts[1] ||
                std::filesystem::file_size(argv[6])!=8+uint64_t(counts[0])*20+uint64_t(counts[1])*12)
                throw std::runtime_error("Invalid packed UV dump");
            bm.T=atlas;bm.verts.resize(size_t(counts[0])*3);bm.uv.resize(size_t(counts[0])*2);
            bm.faces.resize(size_t(counts[1])*3);
            packed.read(reinterpret_cast<char*>(bm.verts.data()),bm.verts.size()*4);
            packed.read(reinterpret_cast<char*>(bm.uv.data()),bm.uv.size()*4);
            packed.read(reinterpret_cast<char*>(bm.faces.data()),bm.faces.size()*4);
            if (!packed) throw std::runtime_error("Truncated packed UV dump");
        } else if (mode == "chart" && argc==6) bm = trellis::uv_chart_project(vertices, int(quad.p.size()), triangles, int(triangles.size()/3), {}, atlas, &vox);
        else if (mode == "box" && argc==6) bm = trellis::uv_box_project(vertices, int(quad.p.size()), triangles, int(triangles.size()/3), {}, atlas, &vox);
        else if (mode == "xatlas" && argc==6) bm = trellis::uv_bake(vertices, int(quad.p.size()), triangles, int(triangles.size()/3), {}, atlas, &vox);
        else throw std::runtime_error("Invalid UV mode or argument count");
        if (!bm.ok() || bm.faces.size() != triangles.size() || bm.uv.size()*3 != bm.verts.size()*2)
            throw std::runtime_error("Invalid baked mesh");
        std::vector<UvPoint3> baked_positions(bm.verts.size()/3);
        for (size_t i=0; i<baked_positions.size(); ++i)
            for (int j=0; j<3; ++j) baked_positions[i][j]=bm.verts[3*i+j];
        std::vector<std::array<uint32_t,3>> baked_faces(bm.faces.size()/3);
        for (size_t i=0; i<baked_faces.size(); ++i)
            for (int j=0; j<3; ++j) {
                const int32_t v = bm.faces[3*i+j];
                if (v < 0 || size_t(v) >= baked_positions.size()) throw std::runtime_error("Invalid baked face index");
                baked_faces[i][j] = uint32_t(v);
            }
        const auto mapping=match_quad_uv_faces(quad, baked_positions, baked_faces);
        if (mode=="packed") preset_stats=trellis::bake_preset_uv(bm,vox);
        if (bm.base.size() != size_t(atlas)*atlas*4 || bm.mr.size() != bm.base.size())
            throw std::runtime_error("Missing baked textures");
        const std::string out(argv[3]);
        {
            std::ofstream uv(out + ".uv.bin", std::ios::binary);
            uint32_t counts[2] = {uint32_t(baked_positions.size()), uint32_t(baked_faces.size())};
            uv.write(reinterpret_cast<const char*>(counts), sizeof(counts));
            uv.write(reinterpret_cast<const char*>(bm.verts.data()), bm.verts.size()*4);
            uv.write(reinterpret_cast<const char*>(bm.uv.data()), bm.uv.size()*4);
            uv.write(reinterpret_cast<const char*>(bm.faces.data()), bm.faces.size()*4);
            if (!uv) throw std::runtime_error("UV dump write failed");
        }
        int64_t seed=-1;
        if (const char* value=std::getenv("TRELLIS_RETOPO_SEED")) {
            size_t used=0;
            const std::string text(value);
            seed=std::stoll(text,&used);
            if (used!=text.size()) throw std::runtime_error("Invalid retopo seed");
        }
        const char* copyright=std::getenv("TRELLIS_RETOPO_COPYRIGHT");
        if (!trellis::write_glb_textured(out.c_str(), bm.verts.data(), int64_t(baked_positions.size()),
                bm.uv.data(), bm.faces.data(), int64_t(baked_faces.size()), bm.base.data(), bm.mr.data(), atlas,
                false, seed, copyright, false)) throw std::runtime_error("GLB write failed");
        if (mode=="packed") write_textured_quads(out,quad,bm,baked_positions,baked_faces,mapping);
        std::cout << "quads=" << quad.q.size() << " baked_triangles=" << baked_faces.size()
                  << " atlas=" << atlas << " uv_vertices=" << baked_positions.size()
                  << " covered_texels=" << preset_stats.covered_texels
                  << " missing_voxel_texels=" << preset_stats.missing_voxel_texels
                  << " projected_samples=" << preset_stats.projected_samples
                  << " projected_samples_above_0_008=" << preset_stats.projected_samples_above_0_008
                  << " maximum_source_distance=" << preset_stats.maximum_source_distance << '\n';
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 2;
    }
}
