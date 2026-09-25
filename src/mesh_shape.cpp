#include "mesh_shape.h"
#include "tri_bvh.h"
#include "triangle_intersection.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace trellis {
namespace {

using V = std::array<double,3>;
using F = std::array<int32_t,3>;

V position(const std::vector<float>& vertices, int32_t id) {
    return {vertices[3*id],vertices[3*id+1],vertices[3*id+2]};
}

V sub(const V& a, const V& b) { return {a[0]-b[0],a[1]-b[1],a[2]-b[2]}; }
V cross(const V& a, const V& b) {
    return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};
}
double dot(const V& a, const V& b) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
double length(const V& a) { return std::sqrt(dot(a,a)); }

double angle_quality(const std::vector<float>& vertices, F face) {
    double result = 4.0;
    for (int i = 0; i < 3; ++i) {
        V u = sub(position(vertices,face[(i+1)%3]),position(vertices,face[i]));
        V v = sub(position(vertices,face[(i+2)%3]),position(vertices,face[i]));
        result = std::min(result,std::atan2(length(cross(u,v)),dot(u,v)));
    }
    return result;
}

double diagonal_gap(const std::vector<float>& vertices, int a, int b, int c, int d) {
    V ab = sub(position(vertices,b),position(vertices,a));
    V cd = sub(position(vertices,d),position(vertices,c));
    V ac = sub(position(vertices,c),position(vertices,a));
    double aa=dot(ab,ab),bb=dot(cd,cd),abdot=dot(ab,cd),ad=dot(ab,ac),bd=dot(cd,ac);
    double det=aa*bb-abdot*abdot;
    if (!(det>1e-12*aa*bb)) return std::numeric_limits<double>::infinity();
    double t=(ad*bb-bd*abdot)/det,u=(ad*abdot-bd*aa)/det;
    if (!(t>0 && t<1 && u>0 && u<1)) return std::numeric_limits<double>::infinity();
    V gap={t*ab[0]-u*cd[0]-ac[0],t*ab[1]-u*cd[1]-ac[1],t*ab[2]-u*cd[2]-ac[2]};
    return length(gap);
}

uint64_t edge_key(int32_t a,int32_t b) {
    if (a>b) std::swap(a,b);
    return (uint64_t(uint32_t(a))<<32)|uint32_t(b);
}

Triangle3 triangle(const std::vector<float>& vertices,F face) {
    return {position(vertices,face[0]),position(vertices,face[1]),position(vertices,face[2])};
}

struct Box { float lo[3],hi[3]; };
Box bounds(const std::vector<float>& vertices,std::array<int32_t,4> ids) {
    Box box{{1e30f,1e30f,1e30f},{-1e30f,-1e30f,-1e30f}};
    for (int id:ids) for (int k=0;k<3;++k) {
        float x=vertices[3*id+k];
        box.lo[k]=std::min(box.lo[k],x);
        box.hi[k]=std::max(box.hi[k],x);
    }
    return box;
}
bool overlap(const Box& a,const Box& b) {
    for (int k=0;k<3;++k) if (a.hi[k]<b.lo[k] || b.hi[k]<a.lo[k]) return false;
    return true;
}

double orient(const V& a,const V& b,const V& c,const V& d) {
    return dot(sub(b,a),cross(sub(c,a),sub(d,a)));
}
bool inside_tetra(const V& a,const V& b,const V& c,const V& d,const V& p) {
    double total=orient(a,b,c,d);
    if (std::abs(total)<1e-18) return false;
    double values[]={orient(p,b,c,d),orient(a,p,c,d),orient(a,b,p,d),orient(a,b,c,p)};
    for (double value:values) if (value*total<=0.0) return false;
    return true;
}

struct Edge { int32_t a,b,c,d;int32_t f,g=-1;int count=1; };
struct Candidate { Edge edge;F first,second;Box box;double gain; };

}

ShapeRepairStats flip_skinny_triangles(const std::vector<float>& vertices,
                                       std::vector<int32_t>& faces,
                                       float minimum_angle_degrees,
                                       float maximum_diagonal_gap,
                                       int rings,int passes) {
    if (vertices.size()%3 || faces.size()%3 || minimum_angle_degrees<=0 ||
        maximum_diagonal_gap<=0 || rings<0 || passes<0)
        throw std::invalid_argument("invalid shape repair arguments");
    const int Fcount=int(faces.size()/3), Vcount=int(vertices.size()/3);
    ShapeRepairStats stats;
    for (int pass=0;pass<passes;++pass) {
        std::unordered_map<uint64_t,Edge> edges;
        edges.reserve(size_t(Fcount)*2);
        std::vector<double> quality(Fcount);
        for (int face=0;face<Fcount;++face) {
            F t={faces[3*face],faces[3*face+1],faces[3*face+2]};
            for (int id:t) if (id<0 || id>=Vcount) throw std::invalid_argument("invalid face index");
            quality[face]=angle_quality(vertices,t);
            for (int i=0;i<3;++i) {
                int32_t a=t[i],b=t[(i+1)%3],c=t[(i+2)%3];
                auto [it,added]=edges.emplace(edge_key(a,b),Edge{a,b,c,-1,face});
                if (!added) {it->second.d=c;it->second.g=face;++it->second.count;}
            }
        }
        std::vector<uint8_t> active(Fcount);
        double threshold=minimum_angle_degrees*0.01745329251994329577;
        for (int f=0;f<Fcount;++f) active[f]=quality[f]<threshold;
        for (int ring=0;ring<rings;++ring) {
            auto next=active;
            for (const auto& item:edges) {
                const Edge& e=item.second;
                if (e.count==2 && (active[e.f] || active[e.g])) next[e.f]=next[e.g]=1;
            }
            active.swap(next);
        }
        std::vector<Candidate> candidates;
        for (const auto& item:edges) {
            const Edge& e=item.second;
            if (e.count!=2 || !active[e.f] || !active[e.g] || e.c==e.d ||
                edges.count(edge_key(e.c,e.d))) continue;
            F first={e.c,e.d,e.b},second={e.d,e.c,e.a};
            double before=std::min(quality[e.f],quality[e.g]);
            double after=std::min(angle_quality(vertices,first),angle_quality(vertices,second));
            if (!(after>before+1e-6) ||
                diagonal_gap(vertices,e.a,e.b,e.c,e.d)>maximum_diagonal_gap) continue;
            candidates.push_back({e,first,second,bounds(vertices,{e.a,e.b,e.c,e.d}),after-before});
        }
        std::sort(candidates.begin(),candidates.end(),[](const Candidate& a,const Candidate& b){return a.gain>b.gain;});
        TriBvh tree=TriBvh::build(vertices.data(),Vcount,faces.data(),Fcount);
        std::vector<uint8_t> used(Vcount);
        std::vector<Box> accepted_boxes;
        std::vector<Candidate> accepted;
        for (const Candidate& candidate:candidates) {
            const Edge& e=candidate.edge;
            if (used[e.a] || used[e.b] || used[e.c] || used[e.d]) continue;
            bool blocked=false;
            for (const Box& box:accepted_boxes) if (overlap(box,candidate.box)) {blocked=true;break;}
            if (blocked) continue;
            struct Query {
                const std::vector<float>* vertices;
                const std::vector<int32_t>* faces;
                const Candidate* candidate;
                bool collision=false,swept=false;
            } query{&vertices,&faces,&candidate};
            const auto visit=[](void* pointer,int32_t face)->bool {
                auto& q=*static_cast<Query*>(pointer);
                if (face==q.candidate->edge.f || face==q.candidate->edge.g) return true;
                F ids={(*q.faces)[3*face],(*q.faces)[3*face+1],(*q.faces)[3*face+2]};
                Triangle3 other=triangle(*q.vertices,ids);
                for (F proposal:{q.candidate->first,q.candidate->second})
                    if (triangles_overlap_interior(inset_triangle(triangle(*q.vertices,proposal)),inset_triangle(other))) {
                        q.collision=true;return false;
                    }
                V a=position(*q.vertices,q.candidate->edge.a),b=position(*q.vertices,q.candidate->edge.b);
                V c=position(*q.vertices,q.candidate->edge.c),d=position(*q.vertices,q.candidate->edge.d);
                for (int id:ids) if (inside_tetra(a,b,c,d,position(*q.vertices,id))) {
                    q.swept=true;return false;
                }
                return true;
            };
            if (!tree.visit_overlapping(candidate.box.lo,candidate.box.hi,visit,&query)) {
                stats.collision_rejections+=query.collision;
                stats.swept_rejections+=query.swept;
                continue;
            }
            used[e.a]=used[e.b]=used[e.c]=used[e.d]=1;
            accepted_boxes.push_back(candidate.box);
            accepted.push_back(candidate);
        }
        if (accepted.empty()) break;
        for (const Candidate& candidate:accepted) {
            const Edge& e=candidate.edge;
            for (int i=0;i<3;++i) {
                faces[3*e.f+i]=candidate.first[i];
                faces[3*e.g+i]=candidate.second[i];
            }
        }
        stats.flips+=int(accepted.size());
    }
    return stats;
}

SmoothingStats smooth_skinny_triangles(std::vector<float>& vertices,
                                      const std::vector<int32_t>& faces,
                                      float minimum_angle_degrees,
                                      float maximum_displacement,int passes) {
    if (vertices.size()%3 || faces.size()%3 || minimum_angle_degrees<=0 ||
        maximum_displacement<=0 || passes<0)
        throw std::invalid_argument("invalid smoothing arguments");
    const int Vcount=int(vertices.size()/3),Fcount=int(faces.size()/3);
    const std::vector<float> original=vertices;
    std::vector<std::vector<int32_t>> incident(Vcount);
    for (int face=0;face<Fcount;++face) for (int j=0;j<3;++j) {
        int id=faces[3*face+j];
        if (id<0 || id>=Vcount) throw std::invalid_argument("invalid face index");
        incident[id].push_back(face);
    }
    const double threshold=minimum_angle_degrees*0.01745329251994329577;
    SmoothingStats stats;
    std::vector<int32_t> face_stamp(Fcount,-1);
    for (int pass=0;pass<passes;++pass) {
        std::vector<std::pair<double,int32_t>> active;
        std::vector<uint8_t> selected(Vcount);
        for (int face=0;face<Fcount;++face) {
            F ids={faces[3*face],faces[3*face+1],faces[3*face+2]};
            double angle=angle_quality(vertices,ids);
            if (angle>=threshold) continue;
            for (int id:ids) if (!selected[id]) {
                selected[id]=1;
                active.emplace_back(angle,id);
            }
        }
        if (active.empty()) break;
        std::sort(active.begin(),active.end());
        TriBvh tree=TriBvh::build(vertices.data(),Vcount,faces.data(),Fcount);
        int accepted=0;
        for (const auto& item:active) {
            const int vertex=item.second;
            const V start=position(vertices,vertex);
            const V origin=position(original,vertex);
            std::vector<double> before;
            std::vector<V> normals;
            double old_min=4.0;
            for (int face:incident[vertex]) {
                F ids={faces[3*face],faces[3*face+1],faces[3*face+2]};
                double angle=angle_quality(vertices,ids);
                before.push_back(angle);
                old_min=std::min(old_min,angle);
                normals.push_back(cross(sub(position(vertices,ids[1]),position(vertices,ids[0])),
                                        sub(position(vertices,ids[2]),position(vertices,ids[0]))));
                face_stamp[face]=vertex;
            }
            if (old_min>=threshold) continue;
            std::vector<V> directions={{{1,0,0}},{{-1,0,0}},{{0,1,0}},{{0,-1,0}},{{0,0,1}},{{0,0,-1}}};
            V centroid{};int neighbors=0;
            for (int face:incident[vertex]) for (int j=0;j<3;++j) {
                int id=faces[3*face+j];
                if (id==vertex) continue;
                V p=position(vertices,id);
                for (int k=0;k<3;++k) centroid[k]+=p[k];
                ++neighbors;
            }
            if (neighbors) {
                V toward={centroid[0]/neighbors-start[0],centroid[1]/neighbors-start[1],centroid[2]/neighbors-start[2]};
                directions.push_back(toward);
                directions.push_back({-toward[0],-toward[1],-toward[2]});
            }
            for (size_t j=0;j<incident[vertex].size();++j) {
                if (before[j]>=threshold) continue;
                F ids={faces[3*incident[vertex][j]],faces[3*incident[vertex][j]+1],faces[3*incident[vertex][j]+2]};
                V gradient{};
                for (int k=0;k<3;++k) {
                    float old=vertices[3*vertex+k];
                    vertices[3*vertex+k]=old+1e-5f;
                    gradient[k]=angle_quality(vertices,ids)-before[j];
                    vertices[3*vertex+k]=old;
                }
                directions.push_back(gradient);
                directions.push_back({-gradient[0],-gradient[1],-gradient[2]});
            }
            V best=start;double best_min=old_min;
            for (const V& direction:directions) {
                double norm=length(direction);
                if (!(norm>1e-15)) continue;
                for (double radius:{0.0001,0.00025,0.0005,0.001}) {
                    V proposal{};
                    for (int k=0;k<3;++k) proposal[k]=start[k]+radius*direction[k]/norm;
                    V offset=sub(proposal,origin);
                    double distance=length(offset);
                    if (distance>maximum_displacement)
                        for (int k=0;k<3;++k) proposal[k]=origin[k]+maximum_displacement*offset[k]/distance;
                    for (int k=0;k<3;++k) vertices[3*vertex+k]=float(proposal[k]);
                    V rounded=position(vertices,vertex);
                    if (rounded==start) continue;
                    double new_min=4.0;bool valid=true;
                    std::vector<Triangle3> proposed;
                    for (size_t j=0;j<incident[vertex].size();++j) {
                        int face=incident[vertex][j];
                        F ids={faces[3*face],faces[3*face+1],faces[3*face+2]};
                        double angle=angle_quality(vertices,ids);
                        new_min=std::min(new_min,angle);
                        V normal=cross(sub(position(vertices,ids[1]),position(vertices,ids[0])),
                                       sub(position(vertices,ids[2]),position(vertices,ids[0])));
                        if (!(angle+1e-6>=std::min(before[j],threshold)) || dot(normal,normals[j])<=0) {
                            valid=false;break;
                        }
                        proposed.push_back(triangle(vertices,ids));
                    }
                    if (valid && new_min>best_min+1e-6) {
                        struct Query {
                            const std::vector<float>* vertices;
                            const std::vector<int32_t>* faces;
                            const std::vector<int32_t>* stamps;
                            int vertex;
                            Triangle3 proposal;
                        } query{&vertices,&faces,&face_stamp,vertex,{}};
                        const auto visit=[](void* pointer,int32_t face)->bool {
                            auto& q=*static_cast<Query*>(pointer);
                            if ((*q.stamps)[face]==q.vertex) return true;
                            F ids={(*q.faces)[3*face],(*q.faces)[3*face+1],(*q.faces)[3*face+2]};
                            return !triangles_overlap_interior(inset_triangle(q.proposal),
                                                               inset_triangle(triangle(*q.vertices,ids)));
                        };
                        bool clear=true;
                        for (const auto& tri:proposed) {
                            float lo[3]={1e30f,1e30f,1e30f},hi[3]={-1e30f,-1e30f,-1e30f};
                            for (const V& p:tri) for (int k=0;k<3;++k) {
                                lo[k]=std::min(lo[k],float(p[k]));hi[k]=std::max(hi[k],float(p[k]));
                            }
                            query.proposal=tri;
                            if (!tree.visit_overlapping(lo,hi,visit,&query)) {clear=false;break;}
                        }
                        if (clear) for (size_t a=0;a<proposed.size();++a)
                            for (size_t b=0;b<a;++b)
                                if (triangles_overlap_interior(inset_triangle(proposed[a]),
                                                               inset_triangle(proposed[b]))) clear=false;
                        if (clear) {best=rounded;best_min=new_min;}
                        else ++stats.collision_rejections;
                    }
                }
            }
            for (int k=0;k<3;++k) vertices[3*vertex+k]=float(best[k]);
            if (best!=start) {
                ++accepted;++stats.moved;
                stats.maximum_displacement=std::max(stats.maximum_displacement,float(length(sub(best,origin))));
                tree=TriBvh::build(vertices.data(),Vcount,faces.data(),Fcount);
            }
        }
        if (!accepted) break;
    }
    return stats;
}

}
