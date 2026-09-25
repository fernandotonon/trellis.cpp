#pragma once
#include <tri_bvh.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace trellis {
struct UVNormalMarginSample {
    size_t texel;
    int32_t face;
    std::array<float,3> weights;
};

// Locate nearest UV-surface samples for empty texels within a bounded margin.
inline std::vector<UVNormalMarginSample> uv_normal_margin_samples(
    const std::vector<float>& uv,const std::vector<int32_t>& faces,int size,float margin_pixels=2) {
    if(size<=0||size>16384||uv.empty()||uv.size()%2||faces.empty()||faces.size()%3||
       !std::isfinite(margin_pixels)||margin_pixels<0||margin_pixels>size)
        throw std::runtime_error("Invalid UV margin dimensions");
    for(auto x:uv)if(!std::isfinite(x)||x<0||x>1)throw std::runtime_error("Invalid margin UV coordinate");
    for(auto x:faces)if(x<0||size_t(x)>=uv.size()/2)throw std::runtime_error("Invalid margin face index");
    std::vector<uint8_t> occupied(size_t(size)*size,0);
    for (size_t f=0;f<faces.size();f+=3) {
        float x[3],y[3];
        for(int j=0;j<3;++j) {auto id=faces[f+j];x[j]=uv[2*id]*size;y[j]=uv[2*id+1]*size;}
        float d=(y[1]-y[2])*(x[0]-x[2])+(x[2]-x[1])*(y[0]-y[2]);
        if(std::abs(d)<1e-9f)continue;
        int x0=std::max(0,int(std::floor(std::min({x[0],x[1],x[2]})))),x1=std::min(size-1,int(std::ceil(std::max({x[0],x[1],x[2]}))));
        int y0=std::max(0,int(std::floor(std::min({y[0],y[1],y[2]})))),y1=std::min(size-1,int(std::ceil(std::max({y[0],y[1],y[2]}))));
        for(int yy=y0;yy<=y1;++yy)for(int xx=x0;xx<=x1;++xx) {
            float a=((y[1]-y[2])*(xx+.5f-x[2])+(x[2]-x[1])*(yy+.5f-y[2]))/d;
            float b=((y[2]-y[0])*(xx+.5f-x[2])+(x[0]-x[2])*(yy+.5f-y[2]))/d;
            if(a>=-.001f&&b>=-.001f&&1-a-b>=-.001f)occupied[yy*size+xx]=1;
        }
    }
    std::vector<float> uv_positions(uv.size()/2*3,0);
    for(size_t i=0;i<uv.size()/2;++i)for(int k=0;k<2;++k)uv_positions[3*i+k]=uv[2*i+k];
    auto uv_bvh=TriBvh::build(uv_positions.data(),uv_positions.size()/3,faces.data(),faces.size()/3);
    auto barycentric=[&](const float* p,const int32_t* id) {
        float x[2],y[2],z[2];
        for(int k=0;k<2;++k){x[k]=uv[2*id[1]+k]-uv[2*id[0]+k];y[k]=uv[2*id[2]+k]-uv[2*id[0]+k];z[k]=p[k]-uv[2*id[0]+k];}
        auto dot2=[](const float *a,const float *b){return a[0]*b[0]+a[1]*b[1];};
        double xx=dot2(x,x),yy=dot2(y,y),xy=dot2(x,y),xz=dot2(x,z),yz=dot2(y,z),d=xx*yy-xy*xy;
        if(std::abs(d)<1e-30)return std::array<float,3>{1,0,0};
        float v=(yy*xz-xy*yz)/d,w=(xx*yz-xy*xz)/d;
        return std::array<float,3>{1-v-w,v,w};
    };
    const float margin=margin_pixels/size;std::vector<UVNormalMarginSample> result;
    for(size_t pixel=0;pixel<occupied.size();++pixel) {
        if(occupied[pixel])continue;
        const float query[3]={(float(pixel%size)+.5f)/size,(float(pixel/size)+.5f)/size,0};
        auto uv_hit=uv_bvh.closest(query,margin);
        if(uv_hit.face<0||uv_hit.dist2>margin*margin)continue;
        const auto *id=&faces[3*uv_hit.face];
        auto w=barycentric(uv_hit.point,id);
        float sum=0;for(auto &a:w){a=std::clamp(a,0.f,1.f);sum+=a;}
        if(!(sum>0))throw std::runtime_error("Invalid margin barycentrics");
        for(auto &a:w)a/=sum;
        result.push_back({pixel,uv_hit.face,w});
    }
    return result;
}
}
