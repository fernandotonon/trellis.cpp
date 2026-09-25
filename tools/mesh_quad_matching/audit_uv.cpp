#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/box_intersection_d.h>
#include <CGAL/Box_intersection_d/Box_with_handle_d.h>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>
using K=CGAL::Exact_predicates_inexact_constructions_kernel;
using Triangle=K::Triangle_2;
using Box=CGAL::Box_intersection_d::Box_with_handle_d<double,2,size_t,CGAL::Box_intersection_d::ID_EXPLICIT>;

// Convex triangle interiors overlap iff no edge line separates them.
bool interiors_overlap(const Triangle& a,const Triangle& b) {
  for(int swap=0;swap<2;++swap) {
    const auto& x=swap?b:a;const auto& y=swap?a:b;
    const auto orientation=x.orientation();
    for(int edge=0;edge<3;++edge) {
      bool inside=false;
      for(int j=0;j<3;++j)
        inside|=CGAL::orientation(x[edge],x[(edge+1)%3],y[j])==orientation;
      if(!inside) return false;
    }
  }
  return true;
}

int main(int argc,char** argv) {
  if(argc!=3) return 2;
  try {
    std::ifstream input(argv[1],std::ios::binary);
    uint32_t n[2];input.read(reinterpret_cast<char*>(n),sizeof(n));
    if(!input || !n[0] || !n[1] || std::filesystem::file_size(argv[1])!=8+uint64_t(n[0])*20+uint64_t(n[1])*12)
      throw std::runtime_error("Invalid UV dump");
    std::vector<std::array<float,3>> p(n[0]);
    std::vector<std::array<float,2>> uv(n[0]);
    std::vector<std::array<int32_t,3>> faces(n[1]);
    input.read(reinterpret_cast<char*>(p.data()),p.size()*12);
    input.read(reinterpret_cast<char*>(uv.data()),uv.size()*8);
    input.read(reinterpret_cast<char*>(faces.data()),faces.size()*12);
    for(auto v:p) for(auto x:v) if(!std::isfinite(x)) throw std::runtime_error("Nonfinite position");
    for(auto v:uv) for(auto x:v) if(!std::isfinite(x)) throw std::runtime_error("Nonfinite UV");
    std::vector<Triangle> tris;tris.reserve(n[1]);
    std::vector<Box> boxes;boxes.reserve(n[1]);
    std::ofstream tiny(std::string(argv[2])+".tiny.tsv"),pairs(std::string(argv[2])+".pairs.tsv");
    tiny<<std::setprecision(17)<<"face\tuv_area\tworld_area\tuv_exact_degenerate\n";
    size_t exact_degenerate=0,small=0,outside=0,overlap_pairs=0;
    double total_area=0,small_area=0;
    for(size_t i=0;i<faces.size();++i) {
      auto f=faces[i];
      for(auto x:f) if(x<0 || size_t(x)>=p.size()) throw std::runtime_error("Invalid face index");
      Triangle t(K::Point_2(uv[f[0]][0],uv[f[0]][1]),K::Point_2(uv[f[1]][0],uv[f[1]][1]),K::Point_2(uv[f[2]][0],uv[f[2]][1]));
      tris.push_back(t);
      const bool degenerate=t.is_degenerate();exact_degenerate+=degenerate;
      if(!degenerate) boxes.emplace_back(t.bbox(),i);
      double e[3],d[3];for(int k=0;k<3;++k){e[k]=double(p[f[1]][k])-p[f[0]][k];d[k]=double(p[f[2]][k])-p[f[0]][k];}
      double cross[3]={e[1]*d[2]-e[2]*d[1],e[2]*d[0]-e[0]*d[2],e[0]*d[1]-e[1]*d[0]};
      const double area=.5*std::sqrt(cross[0]*cross[0]+cross[1]*cross[1]+cross[2]*cross[2]);total_area+=area;
      const double u=std::abs(t.area());
      if(u<1e-12) {++small;small_area+=area;tiny<<i<<'\t'<<u<<'\t'<<area<<'\t'<<degenerate<<'\n';}
      for(auto v:f) for(auto x:uv[v]) if(x<0 || x>1) ++outside;
    }
    std::vector<bool> affected(n[1],false);
    CGAL::box_self_intersection_d(boxes.begin(),boxes.end(),[&](const Box& a,const Box& b){
      if(interiors_overlap(tris[a.handle()],tris[b.handle()])) {
        ++overlap_pairs;affected[a.handle()]=affected[b.handle()]=true;
        pairs<<a.handle()<<'\t'<<b.handle()<<'\n';
      }
    });
    size_t affected_count=0;for(bool x:affected) affected_count+=x;
    std::ofstream report(argv[2]);
    report<<std::setprecision(17)<<"{\"faces\":"<<n[1]<<",\"exact_degenerate_uv_faces\":"<<exact_degenerate
      <<",\"below_threshold_uv_faces\":"<<small<<",\"below_threshold_world_area\":"<<small_area
      <<",\"total_world_area\":"<<total_area<<",\"out_of_range_uv_corners\":"<<outside
      <<",\"positive_area_overlap_pairs\":"<<overlap_pairs<<",\"overlapping_faces\":"<<affected_count
      <<",\"accepted\":"<<((!small&&!outside&&!overlap_pairs)?"true":"false")<<"}\n";
    if(!report || !tiny || !pairs) throw std::runtime_error("Cannot write UV audit");
    std::cout<<"faces="<<n[1]<<" exact_degenerate="<<exact_degenerate<<" below_threshold="<<small<<" overlap_pairs="<<overlap_pairs<<" overlapping_faces="<<affected_count<<'\n';
    return (small||outside||overlap_pairs)?1:0;
  } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 2;}
}
