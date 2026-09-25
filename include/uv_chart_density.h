#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace trellis {
class UVChartDensity {
    std::vector<uint32_t> face_chart_;
    std::vector<double> uv_area_, error_, sampled_area_;
    uint32_t chart_count_=0;
public:
    UVChartDensity(const std::vector<std::array<float,2>>& uv,
                   const std::vector<std::array<uint32_t,3>>& faces) {
        if(uv.empty()||faces.empty()||uv.size()>std::numeric_limits<uint32_t>::max())
            throw std::runtime_error("Empty or oversized density mesh");
        for(auto p:uv)for(auto v:p)if(!std::isfinite(v))throw std::runtime_error("Nonfinite UV coordinate");
        std::vector<uint32_t> parent(uv.size());std::iota(parent.begin(),parent.end(),0);
        auto root=[&](uint32_t i) {while(parent[i]!=i){parent[i]=parent[parent[i]];i=parent[i];}return i;};
        for(auto face:faces) {
            for(auto i:face)if(i>=uv.size())throw std::runtime_error("Invalid density face index");
            for(int k=1;k<3;++k)parent[root(face[k])]=root(face[0]);
        }
        std::vector<uint32_t> compact(uv.size(),std::numeric_limits<uint32_t>::max());
        for(auto face:faces) {
            auto r=root(face[0]);
            if(compact[r]==std::numeric_limits<uint32_t>::max())compact[r]=chart_count_++;
            face_chart_.push_back(compact[r]);
        }
        uv_area_.assign(chart_count_,0);error_.assign(chart_count_,0);sampled_area_.assign(faces.size(),0);
        for(size_t i=0;i<faces.size();++i) {
            auto f=faces[i];auto a=uv[f[0]],b=uv[f[1]],c=uv[f[2]];
            double area=.5*std::abs((double(b[0])-a[0])*(double(c[1])-a[1])-(double(b[1])-a[1])*(double(c[0])-a[0]));
            if(!(area>0)||!std::isfinite(area))throw std::runtime_error("Degenerate density UV face");
            uv_area_[face_chart_[i]]+=area;
        }
    }
    void add_sample(size_t face,double area_weight,const std::array<double,5>& absolute_error) {
        if(face>=face_chart_.size()||!std::isfinite(area_weight)||area_weight<=0)
            throw std::runtime_error("Invalid density sample face or area");
        double maximum=0;
        for(auto e:absolute_error) {
            if(!std::isfinite(e)||e<0||e>1.000001)throw std::runtime_error("Invalid material error");
            maximum=std::max(maximum,e);
        }
        error_[face_chart_[face]]+=area_weight*std::max(maximum-1./255,0.);
        sampled_area_[face]+=area_weight;
    }
    std::vector<double> face_scales() const {
        for(auto a:sampled_area_)if(!(a>0)||!std::isfinite(a))throw std::runtime_error("Incomplete density sample coverage");
        double area=std::accumulate(uv_area_.begin(),uv_area_.end(),0.);
        double error=std::accumulate(error_.begin(),error_.end(),0.);
        if(!std::isfinite(area)||!std::isfinite(error))throw std::runtime_error("Density accumulation overflow");
        std::vector<double> relative(chart_count_,1);
        if(error>0)for(uint32_t i=0;i<chart_count_;++i)
            relative[i]=std::clamp(std::cbrt((error_[i]/error)/(uv_area_[i]/area)),.5,2.);
        const double minimum=*std::min_element(relative.begin(),relative.end());
        std::vector<double> result;result.reserve(face_chart_.size());
        for(auto chart:face_chart_)result.push_back(relative[chart]/minimum);
        return result;
    }
    uint32_t chart_count() const {return chart_count_;}
};
}
