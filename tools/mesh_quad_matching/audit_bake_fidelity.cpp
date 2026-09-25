#include "voxel_pbr_sampler.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <cstdlib>
using namespace trellis;
struct Image {
    int width=0,height=0;
    std::unique_ptr<unsigned char,decltype(&stbi_image_free)> data{nullptr,stbi_image_free};
    explicit Image(const char *path) {
        int channels;data.reset(stbi_load(path,&width,&height,&channels,4));
        if(!data||width<=0||height<=0)throw std::runtime_error("Invalid texture");
    }
    float sample(float u,float v,int channel) const {
        float x=std::clamp(u*width-.5f,0.f,float(width-1));
        float y=std::clamp((1-v)*height-.5f,0.f,float(height-1));
        int a=int(x),b=int(y);float fx=x-a,fy=y-b,result=0;
        for(int dy=0;dy<2;++dy)for(int dx=0;dx<2;++dx) {
            int xx=std::min(a+dx,width-1),yy=std::min(b+dy,height-1);
            result+=(dx?fx:1-fx)*(dy?fy:1-fy)*data.get()[4*(yy*width+xx)+channel]/255.f;
        }
        return result;
    }
};
template<class T> void read(std::ifstream &input,std::vector<T> &v) {
    input.read(reinterpret_cast<char*>(v.data()),v.size()*sizeof(T));
    if(!input)throw std::runtime_error("Truncated input");
}
int main(int argc,char **argv) {
    if(argc!=8)return 2;
    try {
        std::ifstream source(argv[1],std::ios::binary);int32_t h[4];source.read(reinterpret_cast<char*>(h),16);
        if(!source||h[0]<=0||h[1]<=0||h[2]<=0||h[3]<=0)throw std::runtime_error("Invalid post dump");
        std::vector<float> vertices(size_t(h[0])*3),pbr(size_t(h[2])*6);
        std::vector<int32_t> faces(size_t(h[1])*3);std::vector<std::array<int,3>> coords(h[2]);
        read(source,vertices);read(source,faces);read(source,coords);read(source,pbr);
        for(auto i:faces)if(i<0||i>=h[0])throw std::runtime_error("Invalid source index");
        VoxelPbr vox{&coords,&pbr,h[3],nullptr};detail::VoxSampler sampler(vox);
        auto bvh=TriBvh::build(vertices.data(),h[0],faces.data(),h[1]);
        std::ifstream uvfile(argv[2],std::ios::binary);uint32_t counts[2];uvfile.read(reinterpret_cast<char*>(counts),8);
        if(!uvfile||!counts[0]||!counts[1])throw std::runtime_error("Invalid UV dump");
        std::vector<float> p(size_t(counts[0])*3),uv(size_t(counts[0])*2);std::vector<int32_t> f(size_t(counts[1])*3);
        read(uvfile,p);read(uvfile,uv);read(uvfile,f);
        for(auto i:f)if(i<0||uint32_t(i)>=counts[0])throw std::runtime_error("Invalid target index");
        Image base(argv[3]),roughness(argv[4]),metallic(argv[5]);
        std::ofstream report(argv[6]);if(!report)throw std::runtime_error("Cannot write report");
        report<<std::setprecision(12)<<"face\tsample\tarea_weight\tr\tg\tb\troughness\tmetallic\tstatus\n";
        std::ofstream missing_details;
        if(const char* path=std::getenv("TRELLIS_FIDELITY_MISSING_PATH")) {
            missing_details.open(path);
            if(!missing_details)throw std::runtime_error("Cannot write missing-source details");
            missing_details<<std::setprecision(12)<<"face\tsample\tarea_weight\ttarget_x\ttarget_y\ttarget_z\tsource_x\tsource_y\tsource_z\tdistance\n";
        }
        const bool heldout=std::getenv("TRELLIS_PROBE_FIDELITY_HELDOUT")!=nullptr;
        const std::vector<std::array<float,3>> weights=heldout?
            std::vector<std::array<float,3>>{{.8f,.1f,.1f},{.1f,.8f,.1f},{.1f,.1f,.8f},{.45f,.45f,.1f},{.45f,.1f,.45f},{.1f,.45f,.45f}}:
            std::vector<std::array<float,3>>{{1.f/3,1.f/3,1.f/3},{.6f,.2f,.2f},{.2f,.6f,.2f},{.2f,.2f,.6f}};
        uint64_t samples=0,missing=0;double weighted[5]={},total_area=0;
        for(uint32_t face=0;face<counts[1];++face) {
            const int32_t *id=&f[3*face];double a[3],b[3],cross[3];
            for(int k=0;k<3;++k){a[k]=double(p[3*id[1]+k])-p[3*id[0]+k];b[k]=double(p[3*id[2]+k])-p[3*id[0]+k];}
            for(int k=0;k<3;++k)cross[k]=a[(k+1)%3]*b[(k+2)%3]-a[(k+2)%3]*b[(k+1)%3];
            double area=.5*std::sqrt(cross[0]*cross[0]+cross[1]*cross[1]+cross[2]*cross[2])/weights.size();
            for(size_t s=0;s<weights.size();++s) {
                float point[3]={},u=0,v=0,value[6];
                for(int j=0;j<3;++j){for(int k=0;k<3;++k)point[k]+=weights[s][j]*p[3*id[j]+k];u+=weights[s][j]*uv[2*id[j]];v+=weights[s][j]*uv[2*id[j]+1];}
                auto hit=bvh.closest(point);bool ok=hit.face>=0&&sampler.sample(hit.point,value);++samples;
                report<<face<<'\t'<<s<<'\t'<<area;
                if(!ok){
                    ++missing;
                    if(missing_details)missing_details<<face<<'\t'<<s<<'\t'<<area
                        <<'\t'<<point[0]<<'\t'<<point[1]<<'\t'<<point[2]
                        <<'\t'<<hit.point[0]<<'\t'<<hit.point[1]<<'\t'<<hit.point[2]
                        <<'\t'<<std::sqrt(double(hit.dist2))<<'\n';
                    report<<"\t0\t0\t0\t0\t0\tmissing_source\n";
                    continue;
                }
                float actual[5]={base.sample(u,v,0),base.sample(u,v,1),base.sample(u,v,2),roughness.sample(u,v,0),metallic.sample(u,v,0)};
                int channel[5]={0,1,2,4,3};total_area+=area;
                for(int k=0;k<5;++k){double error=std::abs(actual[k]-value[channel[k]]);weighted[k]+=area*error;report<<'\t'<<error;}
                report<<"\tok\n";
            }
        }
        if(!report)throw std::runtime_error("Cannot finish report");
        std::ofstream summary(argv[7]);summary<<std::setprecision(12)<<"{\"samples\":"<<samples<<",\"missing_source_samples\":"<<missing<<",\"sampled_area\":"<<total_area<<",\"area_weighted_mae\":[";
        for(int k=0;k<5;++k)summary<<(k?",":"")<<(total_area?weighted[k]/total_area:0);
        summary<<"],\"channels\":[\"r\",\"g\",\"b\",\"roughness\",\"metallic\"],\"complete_pipeline_accepted\":false}\n";
        if(!summary)throw std::runtime_error("Cannot write summary");
        std::cout<<"samples="<<samples<<" missing="<<missing<<" area="<<total_area<<'\n';
    }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 2;}
}
