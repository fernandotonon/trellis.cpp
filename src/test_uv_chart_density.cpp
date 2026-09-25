#include "uv_chart_density.h"
#include <functional>
#include <iostream>
#include <limits>
using trellis::UVChartDensity;
void check(bool ok){if(!ok)throw std::runtime_error("Density regression failed");}
void rejects(const std::function<void()>& f){bool rejected=false;try{f();}catch(const std::runtime_error&){rejected=true;}check(rejected);}
int main(){
    std::vector<std::array<float,2>> uv{{0,0},{1,0},{1,1},{0,1},{2,0},{3,0},{3,1},{2,1}};
    std::vector<std::array<uint32_t,3>> faces{{0,1,2},{0,2,3},{4,5,6},{4,6,7}};
    UVChartDensity density(uv,faces);check(density.chart_count()==2);
    for(size_t f=0;f<4;++f)density.add_sample(f,.5,{f<2?.25:0.,0,0,0,0});
    auto s=density.face_scales();check(std::abs(s[0]-2*std::cbrt(2.))<1e-6&&s[0]==s[1]&&s[2]==1&&s[3]==1);
    UVChartDensity equal(uv,faces);for(size_t f=0;f<4;++f)equal.add_sample(f,.5,{.25,0,0,0,0});for(float x:equal.face_scales())check(x==1);
    UVChartDensity zero(uv,faces);for(size_t f=0;f<4;++f)zero.add_sample(f,.5,{0,0,0,0,0});for(float x:zero.face_scales())check(x==1);
    UVChartDensity incomplete(uv,faces);incomplete.add_sample(0,.5,{.25,0,0,0,0});rejects([&]{incomplete.face_scales();});
    rejects([&]{density.add_sample(4,.5,{0,0,0,0,0});});
    rejects([&]{density.add_sample(0,-1,{0,0,0,0,0});});
    rejects([&]{density.add_sample(0,.5,{std::numeric_limits<double>::quiet_NaN(),0,0,0,0});});
    rejects([&]{density.add_sample(0,.5,{-1,0,0,0,0});});
    auto bad=faces;bad[0][0]=99;rejects([&]{UVChartDensity d(uv,bad);});
    auto collapsed=uv;collapsed[1]=collapsed[0];rejects([&]{UVChartDensity d(collapsed,faces);});
    auto nonfinite=uv;nonfinite[0][0]=std::numeric_limits<float>::infinity();rejects([&]{UVChartDensity d(nonfinite,faces);});
    std::cout<<"PASS allocation oracle, shared chart continuity, equal/zero errors, incomplete coverage and invalid data\n";
}
