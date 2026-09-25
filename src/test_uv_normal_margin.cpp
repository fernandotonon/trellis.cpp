#include "uv_normal_margin.h"
#include <iostream>
#include <limits>
using namespace trellis;
static void check(bool b,const char *msg){if(!b)throw std::runtime_error(msg);}
int main(){try {
    const std::vector<float> uv={.1f,.1f,.25f,.1f,.25f,.25f,.1f,.25f,.65f,.65f,.8f,.65f,.8f,.8f,.65f,.8f};
    const std::vector<int32_t> faces={0,1,2,0,2,3,4,5,6,4,6,7};
    auto samples=uv_normal_margin_samples(uv,faces,16);check(!samples.empty(),"No margin samples");
    bool seen[2]={false,false};
    for(const auto &s:samples) {
        double qx=(s.texel%16+.5)/16,qy=(s.texel/16+.5)/16;
        double lo[2]={uv[0],uv[8]},hi[2]={uv[2],uv[10]},d[2];
        for(int j=0;j<2;++j){double x=std::clamp(qx,lo[j],hi[j]),y=std::clamp(qy,lo[j],hi[j]);d[j]=(x-qx)*(x-qx)+(y-qy)*(y-qy);}
        int j=d[0]<d[1]?0:1;seen[j]=true;
        check(s.face/2==j,"Margin chose a different island");
        double x=0,y=0,sum=0;for(int k=0;k<3;++k){check(s.weights[k]>=0&&s.weights[k]<=1,"Invalid barycentric weight");sum+=s.weights[k];auto i=faces[3*s.face+k];x+=s.weights[k]*uv[2*i];y+=s.weights[k]*uv[2*i+1];}
        check(std::abs(sum-1)<1e-6&&std::abs(x-std::clamp(qx,lo[j],hi[j]))<1e-6&&std::abs(y-std::clamp(qy,lo[j],hi[j]))<1e-6,"Incorrect nearest UV point");
        check(d[j]>0&&d[j]<=.125*.125+1e-8,"Interior or out-of-margin texel returned");
    }
    check(seen[0]&&seen[1],"Did not cover both islands");
    check(uv_normal_margin_samples(uv,faces,16,0).empty(),"Zero margin produced samples");
    auto rejects=[&](std::vector<float> u,std::vector<int32_t> f,int size,float margin){bool caught=false;try{uv_normal_margin_samples(u,f,size,margin);}catch(const std::runtime_error&){caught=true;}check(caught,"Invalid input accepted");};
    rejects({},faces,16,2);rejects(uv,{},16,2);rejects(uv,{0,1},16,2);rejects(uv,{0,1,8},16,2);rejects(uv,{0,1,-1},16,2);
    rejects(uv,faces,0,2);rejects(uv,faces,16,-1);rejects(uv,faces,16,std::numeric_limits<float>::quiet_NaN());
    auto bad=uv;bad[0]=std::numeric_limits<float>::quiet_NaN();rejects(bad,faces,16,2);
    std::cout<<"PASS nearest UV point oracle, separate islands, margin bounds, zero margin and invalid inputs\n";
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
