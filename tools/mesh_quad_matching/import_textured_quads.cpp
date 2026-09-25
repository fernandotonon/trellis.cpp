#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

int main(int argc,char** argv) {
  if(argc!=3)return 2;
  try {
    tinyobj::attrib_t a;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string error;
    const auto base_dir=std::filesystem::path(argv[1]).parent_path().string()+"/";
    if(!tinyobj::LoadObj(&a,&shapes,&materials,&error,argv[1],base_dir.c_str(),false))throw std::runtime_error(error);
    std::vector<std::array<float,3>> p,n;
    std::vector<std::array<float,2>> uv;
    std::vector<std::array<uint32_t,3>> triangles;
    size_t quads=0;
    for(const auto& shape:shapes) {
      size_t offset=0;
      for(size_t i=0;i<shape.mesh.num_face_vertices.size();++i) {
        if(shape.mesh.num_face_vertices[i]!=4)throw std::runtime_error("Importer did not retain a quad");
        const auto material=shape.mesh.material_ids.at(i);
        if(material<0 || size_t(material)>=materials.size())throw std::runtime_error("Missing material assignment");
        const auto& m=materials[material];
        for(const auto& name:{m.diffuse_texname,m.roughness_texname,m.metallic_texname})
          if(name.empty()||!std::filesystem::is_regular_file(std::filesystem::path(argv[1]).parent_path()/name))throw std::runtime_error("Missing PBR texture reference");
        if(!m.normal_texname.empty()&&!std::filesystem::is_regular_file(std::filesystem::path(argv[1]).parent_path()/m.normal_texname))throw std::runtime_error("Missing normal texture reference");
        const uint32_t base=p.size();
        for(size_t j=0;j<4;++j) {
          const auto index=shape.mesh.indices.at(offset+j);
          if(index.vertex_index<0 || index.normal_index<0 || index.texcoord_index<0)throw std::runtime_error("Incomplete OBJ corner");
          p.push_back({a.vertices.at(3*index.vertex_index),a.vertices.at(3*index.vertex_index+1),a.vertices.at(3*index.vertex_index+2)});
          n.push_back({a.normals.at(3*index.normal_index),a.normals.at(3*index.normal_index+1),a.normals.at(3*index.normal_index+2)});
          uv.push_back({a.texcoords.at(2*index.texcoord_index),a.texcoords.at(2*index.texcoord_index+1)});
        }
        triangles.push_back({base,base+1,base+2});triangles.push_back({base,base+2,base+3});
        ++quads;offset+=4;
      }
    }
    if(!quads)throw std::runtime_error("No quads imported");
    std::ofstream out(argv[2],std::ios::binary);const uint32_t counts[2]={uint32_t(p.size()),uint32_t(triangles.size())};
    out.write(reinterpret_cast<const char*>(counts),8);out.write(reinterpret_cast<const char*>(p.data()),p.size()*12);
    out.write(reinterpret_cast<const char*>(uv.data()),uv.size()*8);out.write(reinterpret_cast<const char*>(triangles.data()),triangles.size()*12);
    std::ofstream normals(std::string(argv[2])+".normals.bin",std::ios::binary);normals.write(reinterpret_cast<const char*>(n.data()),n.size()*12);
    if(!out||!normals)throw std::runtime_error("Cannot write imported data");
    std::cout<<"IMPORTED quads="<<quads<<" all_corners_have_uv_and_normal=true all_pbr_texture_references_resolve=true\n";
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
