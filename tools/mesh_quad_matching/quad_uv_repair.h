#pragma once
#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using UvPoint3=std::array<float,3>;
using UvFaceKey=std::array<UvPoint3,3>;
struct UvQuads {std::vector<UvPoint3> p;std::vector<std::array<uint32_t,4>> q;};

inline UvQuads read_uv_quads(const std::string& path) {
  UvQuads mesh;std::ifstream input(path);std::string line;
  while(std::getline(input,line)) {
    std::istringstream row(line);std::string type;row>>type;
    if(type=="v") {
      UvPoint3 p;for(auto& x:p) if(!(row>>x)||!std::isfinite(x)) throw std::runtime_error("Invalid quad position");
      mesh.p.push_back(p);
    } else if(type=="f") {
      std::array<uint32_t,4> q;int64_t v;
      for(auto& x:q) {if(!(row>>v)||v<1||uint64_t(v)>mesh.p.size()) throw std::runtime_error("Invalid quad index");x=v-1;}
      if(row>>type) throw std::runtime_error("Expected four quad corners");
      mesh.q.push_back(q);
    }
  }
  if(!input.eof()||mesh.p.empty()||mesh.q.empty()) throw std::runtime_error("Cannot read quad mesh");
  return mesh;
}

inline UvFaceKey uv_face_key(const std::vector<UvPoint3>& p,const std::array<uint32_t,3>& f) {
  UvFaceKey k={p.at(f[0]),p.at(f[1]),p.at(f[2])};std::sort(k.begin(),k.end());return k;
}

inline std::vector<std::array<uint32_t,2>> match_quad_uv_faces(const UvQuads& mesh,
    const std::vector<UvPoint3>& p,const std::vector<std::array<uint32_t,3>>& f) {
  if(f.size()!=mesh.q.size()*2) throw std::runtime_error("Quad and UV face counts differ");
  std::map<UvFaceKey,uint32_t> index;
  for(uint32_t i=0;i<f.size();++i)
    if(!index.emplace(uv_face_key(p,f[i]),i).second) throw std::runtime_error("Ambiguous UV triangle geometry");
  std::vector<std::array<uint32_t,2>> result;
  for(auto q:mesh.q) {
    std::array<uint32_t,2> pair;int n=0;
    for(auto tri:{std::array<uint32_t,3>{q[0],q[1],q[2]},std::array<uint32_t,3>{q[0],q[2],q[3]}}) {
      auto it=index.find(uv_face_key(mesh.p,tri));
      if(it==index.end()) throw std::runtime_error("Missing quad triangle in UV mesh");
      bool same=false;
      for(int j=0;j<3;++j)
        same|=p[f[it->second][j]]==mesh.p[tri[0]]&&p[f[it->second][(j+1)%3]]==mesh.p[tri[1]]&&p[f[it->second][(j+2)%3]]==mesh.p[tri[2]];
      if(!same) throw std::runtime_error("Quad and UV winding differ");
      pair[n++]=it->second;index.erase(it);
    }
    result.push_back(pair);
  }
  if(!index.empty()) throw std::runtime_error("Unexpected UV geometry");
  return result;
}

inline uint32_t uv_vertex_at(const std::vector<UvPoint3>& p,const std::array<uint32_t,3>& f,const UvPoint3& point) {
  for(auto v:f) if(p.at(v)==point) return v;
  throw std::runtime_error("Missing quad UV corner");
}

// Orthogonal projection gives one continuous UV polygon for a convex quad.
inline size_t repair_quad_uv(const std::string& path,std::vector<UvPoint3>& p,
    std::vector<std::array<float,2>>& uv,std::vector<std::array<uint32_t,3>>& f,
    std::vector<uint32_t>& materials,const std::set<uint32_t>& selected) {
  const auto mesh=read_uv_quads(path);const auto mapping=match_quad_uv_faces(mesh,p,f);
  using D=std::array<double,3>;
  auto cross=[](D a,D b){return D{a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};};
  auto dot=[](D a,D b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];};
  auto unit=[&](D a){double l=std::sqrt(dot(a,a));if(!(l>0)) throw std::runtime_error("Degenerate quad projection");for(auto& x:a)x/=l;return a;};
  size_t repaired=0;
  for(size_t i=0;i<mesh.q.size();++i) {
    auto q=mesh.q[i];auto pair=mapping[i];bool seam=false;
    for(int corner:{0,2}) seam|=uv[uv_vertex_at(p,f[pair[0]],mesh.p[q[corner]])]!=uv[uv_vertex_at(p,f[pair[1]],mesh.p[q[corner]])];
    if(!seam&&!selected.count(pair[0])&&!selected.count(pair[1])) continue;
    std::array<D,4> relative;
    for(int j=0;j<4;++j)for(int k=0;k<3;++k)relative[j][k]=double(mesh.p[q[j]][k])-mesh.p[q[0]][k];
    D normal{};for(int j=0;j<4;++j){auto c=cross(relative[j],relative[(j+1)%4]);for(int k=0;k<3;++k)normal[k]+=c[k];}
    normal=unit(normal);D axis=std::abs(normal[0])<.8?D{1,0,0}:D{0,1,0};
    auto u=unit(cross(axis,normal));auto v=cross(normal,u);
    std::array<std::array<double,2>,4> xy;
    for(int j=0;j<4;++j)xy[j]={dot(relative[j],u),dot(relative[j],v)};
    auto area=[&](int a,int b,int c){return ((xy[b][0]-xy[a][0])*(xy[c][1]-xy[a][1])-(xy[b][1]-xy[a][1])*(xy[c][0]-xy[a][0]))*.5;};
    for(int j=0;j<4;++j)if(!(area(j,(j+1)%4,(j+2)%4)>0)) throw std::runtime_error("Quad projection is not strictly convex");
    double old_area=0;
    for(auto face:pair) {
      const auto a=uv[f[face][0]],b=uv[f[face][1]],c=uv[f[face][2]];
      old_area+=std::abs((double(b[0])-a[0])*(double(c[1])-a[1])-(double(b[1])-a[1])*(double(c[0])-a[0]))*.5;
    }
    const double scale=std::max(std::sqrt(old_area/(area(0,1,2)+area(0,2,3))),std::sqrt(1e-9/std::min(area(0,1,2),area(0,2,3))));
    std::array<uint32_t,4> added;
    for(int j=0;j<4;++j){added[j]=p.size();p.push_back(mesh.p[q[j]]);uv.push_back({float(xy[j][0]*scale),float(xy[j][1]*scale)});}
    for(auto face:pair) {
      for(auto& index:f[face]) {
        bool found=false;for(int j=0;j<4;++j)if(p[index]==mesh.p[q[j]]){index=added[j];found=true;break;}
        if(!found) throw std::runtime_error("Projection changed quad correspondence");
      }
      materials[face]=i+1;
    }
    ++repaired;
  }
  return repaired;
}
