#include "quad_components.h"
#include "tri_bvh.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace trellis {
namespace {
using Point = std::array<float, 3>;

struct Patch {
    std::array<Point, 3> p;
    std::array<double, 3> d;
};

Point midpoint(const Point& a, const Point& b) {
    Point result{};
    for (int k = 0; k < 3; ++k) result[k] = float((double(a[k]) + b[k]) * .5);
    return result;
}

double distance(const Point& a, const Point& b) {
    double squared = 0;
    for (int k = 0; k < 3; ++k) squared += (double(a[k])-b[k])*(double(a[k])-b[k]);
    return std::sqrt(squared);
}

struct Filter {
    const QuadComponentLabels* labels;
    const std::set<size_t>* keep;
};

bool retained(void* data, int32_t face, const float[3]) {
    const auto& filter = *static_cast<Filter*>(data);
    return filter.keep->count(filter.labels->face_labels[size_t(face)/2]) != 0;
}

struct BoundAudit {
    double maximum_leaf_upper_bound = 0;
    uint64_t queries = 0;
};

BoundAudit certify_reverse_distance(
    const std::vector<Point>& vertices,
    const std::vector<std::array<int32_t,3>>& faces,
    const TriBvh& source,
    double limit,
    int max_depth) {
    std::vector<double> vertex_distance(vertices.size(),-1);
    BoundAudit result;
    auto query = [&](const Point& p) {
        const auto hit=source.closest(p.data());
        ++result.queries;
        if (hit.face<0) throw std::runtime_error("Empty source surface");
        const double d=std::sqrt(double(hit.dist2))+1e-5;
        if (d>limit) throw std::runtime_error("Retained quad component exceeds reverse source-distance limit");
        return d;
    };
    auto visit = [&](auto&& self,const Patch& patch,int depth) -> void {
        double bound=std::numeric_limits<double>::infinity();
        for (int i=0;i<3;++i) {
            double radius=0;
            for (int j=0;j<3;++j)
                radius=std::max(radius,distance(patch.p[i],patch.p[j]));
            bound=std::min(bound,patch.d[i]+radius);
        }
        if (bound<=limit) {
            result.maximum_leaf_upper_bound=std::max(result.maximum_leaf_upper_bound,bound);
            return;
        }
        if (depth==max_depth)
            throw std::runtime_error("Retained quad reverse source-distance bound unresolved");
        const Point a=midpoint(patch.p[0],patch.p[1]);
        const Point b=midpoint(patch.p[1],patch.p[2]);
        const Point c=midpoint(patch.p[2],patch.p[0]);
        const double da=query(a),db=query(b),dc=query(c);
        self(self,Patch{{patch.p[0],a,c},{patch.d[0],da,dc}},depth+1);
        self(self,Patch{{a,patch.p[1],b},{da,patch.d[1],db}},depth+1);
        self(self,Patch{{c,b,patch.p[2]},{dc,db,patch.d[2]}},depth+1);
        self(self,Patch{{a,b,c},{da,db,dc}},depth+1);
    };
    for (const auto& face:faces) {
        Patch patch;
        for (int i=0;i<3;++i) {
            patch.p[i]=vertices[face[i]];
            double& d=vertex_distance[face[i]];
            if (d<0) d=query(patch.p[i]);
            patch.d[i]=d;
        }
        visit(visit,patch,0);
    }
    return result;
}
}

QuadSourceCover select_source_covering_quad_components(
    const std::vector<Point>& source_vertices,
    const std::vector<std::array<int32_t, 3>>& source_faces,
    const std::vector<Point>& quad_vertices,
    const std::vector<std::array<uint32_t, 4>>& quads,
    const QuadComponentLabels& labels,
    double limit,
    int max_depth) {
    if (source_vertices.empty() || source_faces.empty() || quad_vertices.empty() ||
        quads.empty() || labels.face_labels.size() != quads.size() ||
        labels.face_counts.empty() || !(limit > 0) || !std::isfinite(limit) ||
        max_depth < 0 || max_depth > 12 ||
        quad_vertices.size() > size_t(INT32_MAX) || quads.size() > size_t(INT32_MAX)/2)
        throw std::invalid_argument("Invalid source-cover input");
    for (size_t id : labels.face_labels)
        if (id >= labels.face_counts.size())
            throw std::invalid_argument("Quad component label out of range");
    for (const auto& p : source_vertices)
        for (float x : p)
            if (!std::isfinite(x)) throw std::invalid_argument("Nonfinite source vertex");
    for (const auto& f : source_faces)
        for (int32_t v : f)
            if (v < 0 || size_t(v) >= source_vertices.size())
                throw std::invalid_argument("Source index out of range");
    for (const auto& p : quad_vertices)
        for (float x : p)
            if (!std::isfinite(x)) throw std::invalid_argument("Nonfinite quad vertex");
    for (const auto& q : quads)
        for (uint32_t v : q)
            if (v >= quad_vertices.size()) throw std::invalid_argument("Quad index out of range");

    std::vector<std::array<int32_t,3>> triangles, alternate;
    triangles.reserve(quads.size()*2);
    alternate.reserve(quads.size()*2);
    for (const auto& q : quads) {
        triangles.push_back({int32_t(q[0]),int32_t(q[1]),int32_t(q[2])});
        triangles.push_back({int32_t(q[0]),int32_t(q[2]),int32_t(q[3])});
        alternate.push_back({int32_t(q[1]),int32_t(q[2]),int32_t(q[3])});
        alternate.push_back({int32_t(q[1]),int32_t(q[3]),int32_t(q[0])});
    }
    const TriBvh full = TriBvh::build(
        reinterpret_cast<const float*>(quad_vertices.data()),quad_vertices.size(),
        reinterpret_cast<const int32_t*>(triangles.data()),triangles.size());
    const TriBvh full_alt = TriBvh::build(
        reinterpret_cast<const float*>(quad_vertices.data()),quad_vertices.size(),
        reinterpret_cast<const int32_t*>(alternate.data()),alternate.size());
    QuadSourceCover result;
    const auto seed = full.closest(source_vertices[source_faces.front()[0]].data());
    if (seed.face < 0) throw std::runtime_error("Full quad mesh is empty");
    result.keep.insert(labels.face_labels[size_t(seed.face)/2]);
    Filter filter{&labels,&result.keep};
    constexpr double numerical_pad = 1e-5;
    for (;;) {
        ++result.rounds;
        double combined_bound = 0;
        bool added = false;
        for (const TriBvh* tree : {&full,&full_alt}) {
            std::vector<double> vertex_distance(source_vertices.size(),-1);
            std::optional<Point> failure;
            double max_bound = 0;
            auto query = [&](const Point& p) {
                const auto hit = tree->closest_filtered(p.data(),retained,&filter,1e15f);
                ++result.distance_queries;
                const double d = hit.face < 0 ? std::numeric_limits<double>::infinity()
                                                : std::sqrt(double(hit.dist2))+numerical_pad;
                if (d > limit) failure = p;
                return d;
            };
            auto visit = [&](auto&& self, const Patch& patch, int depth) -> void {
                if (failure) return;
                double bound = std::numeric_limits<double>::infinity();
                for (int i = 0; i < 3; ++i) {
                    double radius = 0;
                    for (int j = 0; j < 3; ++j)
                        radius = std::max(radius,distance(patch.p[i],patch.p[j]));
                    bound = std::min(bound,patch.d[i]+radius);
                }
                if (bound <= limit) {
                    max_bound = std::max(max_bound,bound);
                    return;
                }
                if (depth == max_depth) {
                    Point centroid{};
                    for (int k = 0; k < 3; ++k)
                        centroid[k] = float((double(patch.p[0][k])+patch.p[1][k]+patch.p[2][k])/3);
                    failure = centroid;
                    return;
                }
                const Point a=midpoint(patch.p[0],patch.p[1]);
                const Point b=midpoint(patch.p[1],patch.p[2]);
                const Point c=midpoint(patch.p[2],patch.p[0]);
                const double da=query(a);
                if (failure) return;
                const double db=query(b);
                if (failure) return;
                const double dc=query(c);
                if (failure) return;
                self(self,Patch{{patch.p[0],a,c},{patch.d[0],da,dc}},depth+1);
                self(self,Patch{{a,patch.p[1],b},{da,patch.d[1],db}},depth+1);
                self(self,Patch{{c,b,patch.p[2]},{dc,db,patch.d[2]}},depth+1);
                self(self,Patch{{a,b,c},{da,db,dc}},depth+1);
            };
            for (const auto& face : source_faces) {
                Patch patch;
                for (int i = 0; i < 3; ++i) {
                    patch.p[i] = source_vertices[face[i]];
                    double& d = vertex_distance[face[i]];
                    if (d < 0) d = query(patch.p[i]);
                    if (failure) break;
                    patch.d[i] = d;
                }
                if (failure) break;
                visit(visit,patch,0);
                if (failure) break;
            }
            if (!failure) {
                combined_bound = std::max(combined_bound,max_bound);
                continue;
            }
            const auto hit = tree->closest(failure->data());
            if (hit.face < 0) throw std::runtime_error("Full quad mesh is empty");
            const size_t id = labels.face_labels[size_t(hit.face)/2];
            if (result.keep.count(id))
                throw std::runtime_error("Source-cover selection cannot resolve witness with another component");
            result.keep.insert(id);
            if (result.keep.size() > labels.face_counts.size())
                throw std::runtime_error("Source-cover selection exceeded component count");
            added = true;
            break;
        }
        if (!added) {
            result.maximum_leaf_upper_bound = combined_bound;
            std::vector<std::array<int32_t,3>> selected_fan,selected_alt;
            size_t selected_quads=0;
            for (size_t id:result.keep) selected_quads+=labels.face_counts[id];
            selected_fan.reserve(selected_quads*2);
            selected_alt.reserve(selected_quads*2);
            for (size_t qi=0;qi<quads.size();++qi) if (result.keep.count(labels.face_labels[qi])) {
                selected_fan.push_back(triangles[2*qi]);
                selected_fan.push_back(triangles[2*qi+1]);
                selected_alt.push_back(alternate[2*qi]);
                selected_alt.push_back(alternate[2*qi+1]);
            }
            const auto source_tree=TriBvh::build(
                reinterpret_cast<const float*>(source_vertices.data()),source_vertices.size(),
                reinterpret_cast<const int32_t*>(source_faces.data()),source_faces.size());
            for (const auto* faces:{&selected_fan,&selected_alt}) {
                const auto audit=certify_reverse_distance(
                    quad_vertices,*faces,source_tree,limit,max_depth);
                result.maximum_reverse_leaf_upper_bound=std::max(
                    result.maximum_reverse_leaf_upper_bound,audit.maximum_leaf_upper_bound);
                result.reverse_distance_queries+=audit.queries;
            }
            return result;
        }
    }
}
}
