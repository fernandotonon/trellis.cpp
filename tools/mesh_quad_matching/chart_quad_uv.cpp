#include "uv_bake.h"
#include "quad_uv_repair.h"
#include <iostream>
int main(int argc,char** argv) {
  if(argc!=3 && argc!=4)return 2;
  try {
    int atlas=argc==4?std::stoi(argv[3]):1024;
    if(atlas!=1024 && atlas!=2048 && atlas!=4096)
      throw std::runtime_error("Invalid atlas resolution");
    auto q=read_uv_quads(argv[1]);
    std::vector<float> verts;verts.reserve(q.p.size()*3);
    for(auto p:q.p) for(auto x:p) verts.push_back(x);
    std::vector<int32_t> f;f.reserve(q.q.size()*6);
    for(auto face:q.q) for(auto tri:{std::array<uint32_t,3>{face[0],face[1],face[2]},
                                    std::array<uint32_t,3>{face[0],face[2],face[3]}})
      for(auto index:tri) f.push_back(int32_t(index));
    auto b=trellis::uv_chart_project(verts,verts.size()/3,f,f.size()/3,{},atlas,nullptr);
    if(!b.ok())throw std::runtime_error("Chart projection failed");
    std::vector<UvPoint3> p(b.verts.size()/3);std::vector<std::array<uint32_t,3>> faces(b.faces.size()/3);
    for(size_t i=0;i<p.size();++i)for(int k=0;k<3;++k)p[i][k]=b.verts[3*i+k];
    for(size_t i=0;i<faces.size();++i)for(int k=0;k<3;++k)faces[i][k]=b.faces[3*i+k];
    match_quad_uv_faces(read_uv_quads(argv[1]),p,faces);
    std::ofstream out(argv[2],std::ios::binary);uint32_t counts[2]={uint32_t(p.size()),uint32_t(faces.size())};
    out.write(reinterpret_cast<const char*>(counts),8);out.write(reinterpret_cast<const char*>(b.verts.data()),b.verts.size()*4);
    out.write(reinterpret_cast<const char*>(b.uv.data()),b.uv.size()*4);out.write(reinterpret_cast<const char*>(b.faces.data()),b.faces.size()*4);
    if(!out)throw std::runtime_error("Cannot write UV dump");
    std::cout<<"CHART triangles="<<faces.size()<<" exact_geometry_and_winding_preserved=true\n";
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
