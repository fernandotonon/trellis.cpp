#include "xatlas.h"
#include "quad_uv_repair.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <stdexcept>
#include <vector>
#include <cstdlib>
#include <numeric>

int main(int argc,char** argv) {
  if(argc!=4 && argc!=5) return 2;
  try {
    int resolution=1024;
    if(const char* value=std::getenv("TRELLIS_PROBE_ATLAS_RES")) {
      const std::string text(value);size_t end=0;
      resolution=std::stoi(text,&end);
      if(end!=text.size() || (resolution!=1024 && resolution!=2048 && resolution!=4096))
        throw std::runtime_error("Invalid atlas resolution");
    }
    int padding=2;
    if(const char* value=std::getenv("TRELLIS_PROBE_ATLAS_PADDING")) {
      const std::string text(value);size_t end=0;
      padding=std::stoi(text,&end);
      if(end!=text.size() || (padding!=1 && padding!=2))
        throw std::runtime_error("Invalid atlas padding");
    }
    std::ifstream in(argv[1],std::ios::binary);
    uint32_t n[2];in.read(reinterpret_cast<char*>(n),8);
    if(!in || !n[0] || !n[1] || std::filesystem::file_size(argv[1])!=8+uint64_t(n[0])*20+uint64_t(n[1])*12)
      throw std::runtime_error("Invalid UV dump");
    std::vector<std::array<float,3>> p(n[0]);
    std::vector<std::array<float,2>> uv(n[0]);
    std::vector<std::array<uint32_t,3>> f(n[1]);
    in.read(reinterpret_cast<char*>(p.data()),p.size()*12);
    in.read(reinterpret_cast<char*>(uv.data()),uv.size()*8);
    in.read(reinterpret_cast<char*>(f.data()),f.size()*12);
    for(auto v:p) for(auto x:v) if(!std::isfinite(x)) throw std::runtime_error("Nonfinite position");
    for(auto v:uv) for(auto x:v) if(!std::isfinite(x)) throw std::runtime_error("Nonfinite UV");
    for(auto t:f) for(auto x:t) if(x>=p.size()) throw std::runtime_error("Invalid vertex index");
    std::ifstream selected(argv[3]);std::set<uint32_t> ids;int64_t id;
    while(selected>>id) {
      if(id<0 || uint64_t(id)>=f.size()) throw std::runtime_error("Invalid selected face");
      ids.insert(uint32_t(id));
    }
    if(!selected.eof()) throw std::runtime_error("Cannot read selected faces");
    std::vector<uint32_t> materials(f.size(),0);
    const size_t repaired_quads=argc==5?repair_quad_uv(argv[4],p,uv,f,materials,ids):0;
    if(argc==4) for(auto i:ids) {
      auto face=f[i];
      double x[3],y[3];
      for(int j=0;j<3;++j) {x[j]=double(uv[face[j]][0])-uv[face[0]][0];y[j]=double(uv[face[j]][1])-uv[face[0]][1];}
      const double area=std::abs(x[1]*y[2]-x[2]*y[1])*.5;
      if(area<=0) throw std::runtime_error("Cannot repack an exactly collapsed triangle");
      const double scale=std::max(1.,std::sqrt(1e-9/area));
      for(int j=0;j<3;++j) {
        f[i][j]=uint32_t(p.size());
        p.push_back(p[face[j]]);
        uv.push_back({float(x[j]*scale),float(y[j]*scale)});
      }
      materials[i]=i+1;
    }
    std::vector<int64_t> remap(p.size(),-1);
    std::vector<std::array<float,3>> compact_p;
    std::vector<std::array<float,2>> compact_uv;
    for(auto& face:f) for(auto& v:face) {
      if(remap[v]<0) {
        remap[v]=compact_p.size();compact_p.push_back(p[v]);compact_uv.push_back(uv[v]);
      }
      v=uint32_t(remap[v]);
    }
    p=std::move(compact_p);uv=std::move(compact_uv);
    if (const char *path=std::getenv("TRELLIS_PROBE_UV_SCALES")) {
      std::vector<uint32_t> parent(p.size());std::iota(parent.begin(),parent.end(),0);
      auto root=[&](uint32_t v) {
        while(parent[v]!=v) {parent[v]=parent[parent[v]];v=parent[v];}
        return v;
      };
      for(auto face:f) for(int k=1;k<3;++k) parent[root(face[k])]=root(face[0]);
      std::vector<float> scale(p.size(),1);
      std::ifstream weights(path);
      for(auto face:f) {
        float s;
        if(!(weights>>s)||!std::isfinite(s)||s<1||s>8) throw std::runtime_error("Invalid per-face UV scale");
        auto r=root(face[0]);scale[r]=std::max(scale[r],s);
      }
      std::string extra;if(weights>>extra) throw std::runtime_error("Extra per-face UV scales");
      const auto original=uv;size_t scaled=0,charts=0;
      for(uint32_t i=0;i<uv.size();++i) {
        auto r=root(i);
        if(i==r) {++charts;scaled+=scale[r]>1;}
        for(int k=0;k<2;++k) uv[i][k]=original[r][k]+scale[r]*(original[i][k]-original[r][k]);
      }
      std::cout<<"DENSITY components="<<charts<<" scaled="<<scaled<<'\n';
    }
    for(auto& v:uv) for(auto& x:v) x*=1e6f;
    std::unique_ptr<xatlas::Atlas,decltype(&xatlas::Destroy)> atlas(xatlas::Create(),xatlas::Destroy);
    xatlas::UvMeshDecl mesh;
    mesh.vertexUvData=uv.data();mesh.vertexCount=uv.size();mesh.vertexStride=8;
    mesh.indexData=f.data();mesh.indexCount=f.size()*3;mesh.indexFormat=xatlas::IndexFormat::UInt32;
    mesh.faceMaterialData=materials.data();
    if(xatlas::AddUvMesh(atlas.get(),mesh)!=xatlas::AddMeshError::Success) throw std::runtime_error("AddUvMesh failed");
    xatlas::PackOptions options;options.resolution=resolution;options.padding=padding;options.bilinear=true;
    options.rotateChartsToAxis=false;
    xatlas::ComputeCharts(atlas.get());
    std::cerr<<"uv_chart_computation_complete"<<std::endl;
    xatlas::PackCharts(atlas.get(),options);
    std::cerr<<"initial_pack atlases="<<atlas->atlasCount<<" width="<<atlas->width
             <<" height="<<atlas->height<<" texels_per_unit="<<atlas->texelsPerUnit<<std::endl;
    bool accepted=atlas->atlasCount==1 && atlas->width==uint32_t(resolution) &&
                  atlas->height==uint32_t(resolution);
    options.texelsPerUnit=atlas->texelsPerUnit;
    if(const char* value=std::getenv("TRELLIS_PROBE_ATLAS_TPU")) {
      char* end=nullptr;
      const float target=std::strtof(value,&end);
      if(end==value || *end || !std::isfinite(target) || target<=0)
        throw std::runtime_error("Invalid atlas texels-per-unit override");
      options.texelsPerUnit=target;
      accepted=false;
    }
    float backoff=.8f;
    if(const char* value=std::getenv("TRELLIS_PROBE_ATLAS_BACKOFF")) {
      char* end=nullptr;
      backoff=std::strtof(value,&end);
      if(end==value || *end || !std::isfinite(backoff) || backoff<=0.5f || backoff>=1.f)
        throw std::runtime_error("Invalid atlas backoff factor");
    }
    if(!accepted) for(int attempt=0;attempt<32;++attempt) {
      xatlas::PackCharts(atlas.get(),options);
      std::cerr<<"pack_attempt="<<attempt<<" atlases="<<atlas->atlasCount
               <<" width="<<atlas->width<<" height="<<atlas->height
               <<" texels_per_unit="<<atlas->texelsPerUnit<<std::endl;
      accepted=atlas->atlasCount==1 && atlas->width==uint32_t(resolution) &&
               atlas->height==uint32_t(resolution);
      if(accepted) break;
      options.texelsPerUnit*=backoff;
    }
    if(!accepted || atlas->meshCount!=1 || atlas->atlasCount!=1 || atlas->width!=uint32_t(resolution) || atlas->height!=uint32_t(resolution))
      throw std::runtime_error("Invalid packed atlas: meshes="+std::to_string(atlas->meshCount)+
        " atlases="+std::to_string(atlas->atlasCount)+" width="+std::to_string(atlas->width)+
        " height="+std::to_string(atlas->height)+" charts="+std::to_string(atlas->chartCount)+
        " texels_per_unit="+std::to_string(atlas->texelsPerUnit));
    const auto& out=atlas->meshes[0];
    if(out.indexCount!=f.size()*3) throw std::runtime_error("Packing changed face count");
    std::vector<std::array<float,3>> op(out.vertexCount);
    std::vector<std::array<float,2>> ou(out.vertexCount);
    std::vector<std::array<uint32_t,3>> of(f.size());
    for(size_t i=0;i<op.size();++i) {
      auto v=out.vertexArray[i];
      if(v.atlasIndex!=0 || v.chartIndex<0 || v.xref>=p.size()) throw std::runtime_error("Packing omitted a vertex");
      op[i]=p[v.xref];ou[i]={v.uv[0]/atlas->width,v.uv[1]/atlas->height};
    }
    for(size_t i=0;i<f.size();++i) for(int j=0;j<3;++j) {
      const auto index=out.indexArray[3*i+j];
      if(index>=op.size() || op[index]!=p[f[i][j]]) throw std::runtime_error("Packing changed face geometry or order");
      of[i][j]=index;
    }
    std::ofstream output(argv[2],std::ios::binary);uint32_t counts[2]={uint32_t(op.size()),uint32_t(of.size())};
    output.write(reinterpret_cast<const char*>(counts),8);
    output.write(reinterpret_cast<const char*>(op.data()),op.size()*12);
    output.write(reinterpret_cast<const char*>(ou.data()),ou.size()*8);
    output.write(reinterpret_cast<const char*>(of.data()),of.size()*12);
    if(!output) throw std::runtime_error("Cannot write repacked UVs");
    std::cout<<"REPACK selected="<<ids.size()<<" repaired_quads="<<repaired_quads<<" charts="<<atlas->chartCount<<" width="<<atlas->width<<" height="<<atlas->height<<" faces="<<of.size()<<" geometry_exact=true\n";
  } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 2;}
}
