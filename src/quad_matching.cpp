#include "quad_matching.h"
#include "quad_quality.h"

#include <lemon/matching.h>
#include <lemon/smart_graph.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>

namespace trellis {

QuadMatchResult match_triangles_to_quads(
    const std::vector<std::array<float, 3>>& vertices,
    const std::vector<std::array<uint32_t, 3>>& triangles,
    const QuadMatchOptions& options) {
    if (vertices.empty() || triangles.empty() ||
        !(options.minimum_scaled_jacobian > 0 && options.minimum_scaled_jacobian <= 1) ||
        !(options.maximum_warpage >= 0) ||
        !(options.minimum_corner_degrees >= 0 && options.minimum_corner_degrees < 90) ||
        !(options.maximum_normal_span_degrees > 0 && options.maximum_normal_span_degrees <= 180) ||
        (!options.face_warpage.empty() && options.face_warpage.size() != triangles.size()))
        throw std::invalid_argument("invalid matching input or options");
    for (const auto& point : vertices)
        for (float value : point)
            if (!std::isfinite(value)) throw std::invalid_argument("nonfinite vertex");
    for (const auto& face : triangles)
        for (uint32_t index : face)
            if (index >= vertices.size()) throw std::invalid_argument("triangle index out of range");
    for (size_t face : options.preferred_faces)
        if (face >= triangles.size()) throw std::invalid_argument("preferred face out of range");
    for (double limit : options.face_warpage)
        if (!(limit >= 0))
            throw std::invalid_argument("invalid face warpage limit");
    for (const auto& pair : options.forbidden_pairs)
        if (pair[0] >= triangles.size() || pair[1] >= triangles.size() || pair[0] >= pair[1])
            throw std::invalid_argument("invalid forbidden pair");

    struct Side { size_t face; uint32_t a, b, c; };
    struct Candidate { size_t f0, f1; std::array<uint32_t, 4> q; double quality; };
    std::map<std::array<uint32_t, 2>, std::vector<Side>> edges;
    for (size_t i = 0; i < triangles.size(); ++i)
        for (int j = 0; j < 3; ++j) {
            const auto a = triangles[i][j], b = triangles[i][(j+1)%3], c = triangles[i][(j+2)%3];
            edges[{std::min(a,b), std::max(a,b)}].push_back({i,a,b,c});
        }
    std::vector<Candidate> candidates;
    candidates.reserve(edges.size());
    for (const auto& entry : edges) {
        const auto& sides = entry.second;
        if (sides.size() != 2) throw std::invalid_argument("expected closed manifold input");
        const auto a = sides[0], b = sides[1];
        if (a.a != b.b || a.b != b.a) throw std::invalid_argument("inconsistent winding");
        const std::array<uint32_t, 4> q = {a.a, b.c, a.b, a.c};
        if (q[0] == q[1] || q[0] == q[2] || q[0] == q[3] ||
            q[1] == q[2] || q[1] == q[3] || q[2] == q[3]) continue;
        std::array<QuadPoint, 4> points;
        for (int corner = 0; corner < 4; ++corner) {
            const auto& p = vertices[q[corner]];
            points[corner] = {p[0],p[1],p[2]};
        }
        candidates.push_back({a.face,b.face,q,quad_min_scaled_jacobian(points)});
    }

    lemon::SmartGraph graph;
    std::vector<lemon::SmartGraph::Node> nodes;
    nodes.reserve(triangles.size());
    for (size_t i = 0; i < triangles.size(); ++i) nodes.push_back(graph.addNode());
    lemon::SmartGraph::EdgeMap<long long> weight(graph);
    lemon::SmartGraph::EdgeMap<size_t> index(graph);
    QuadMatchResult result;
    result.candidate_count = candidates.size();
    result.eligible_degree.assign(triangles.size(), 0);
    result.best_quality.assign(triangles.size(), -1e100);
    const long long face_bonus = static_cast<long long>(triangles.size()) * 1000 + 1;
    const long double bound = static_cast<long double>(face_bonus) *
        (options.preferred_faces.size()+3) * (triangles.size()+1) * 8;
    if (bound > std::numeric_limits<long long>::max())
        throw std::overflow_error("matching weights exceed safe integer range");
    const long long priority = face_bonus * (options.preferred_faces.size()+1);

    auto warpage = [&](const Candidate& c) {
        std::array<double,3> a{}, b{}, d{};
        double aa=0, bb=0, ab=0, ad=0, bd=0;
        for (int j=0; j<3; ++j) {
            a[j] = double(vertices[c.q[2]][j]) - vertices[c.q[0]][j];
            b[j] = double(vertices[c.q[3]][j]) - vertices[c.q[1]][j];
            d[j] = double(vertices[c.q[1]][j]) - vertices[c.q[0]][j];
            aa += a[j]*a[j]; bb += b[j]*b[j]; ab += a[j]*b[j];
            ad += a[j]*d[j]; bd += b[j]*d[j];
        }
        const double det = aa*bb-ab*ab;
        if (!(det > 1e-12*aa*bb)) return std::numeric_limits<double>::infinity();
        const double t=(ad*bb-bd*ab)/det, u=(ad*ab-bd*aa)/det;
        if (!(t>0 && t<1 && u>0 && u<1)) return std::numeric_limits<double>::infinity();
        double distance=0;
        for (int j=0; j<3; ++j) {
            const double delta=t*a[j]-u*b[j]-d[j];
            distance += delta*delta;
        }
        return std::sqrt(distance)+1e-10;
    };
    auto corner_score = [&](const Candidate& c) {
        double minimum=1;
        for (int k=0; k<4; ++k) {
            double aa=0,bb=0,ab=0;
            for (int j=0; j<3; ++j) {
                const double a=double(vertices[c.q[(k+1)%4]][j])-vertices[c.q[k]][j];
                const double b=double(vertices[c.q[(k+3)%4]][j])-vertices[c.q[k]][j];
                aa+=a*a; bb+=b*b; ab+=a*b;
            }
            if (!(aa>0 && bb>0)) return 0.0;
            minimum=std::min(minimum,std::sqrt(std::max(0.0,1-ab*ab/(aa*bb))));
        }
        return minimum;
    };
    auto normal_compatible = [&](const Candidate& c) {
        if (options.maximum_normal_span_degrees == 180) return true;
        std::array<std::array<double,3>,4> normals;
        for (int k=0; k<4; ++k) {
            std::array<double,3> a{},b{};
            for (int j=0; j<3; ++j) {
                a[j]=double(vertices[c.q[(k+1)%4]][j])-vertices[c.q[k]][j];
                b[j]=double(vertices[c.q[(k+3)%4]][j])-vertices[c.q[k]][j];
            }
            auto& n=normals[k];
            n={a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};
            const double length=std::sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]);
            if (!(length>0)) return false;
            for (auto& value:n) value/=length;
        }
        const double limit=std::cos(options.maximum_normal_span_degrees*3.141592653589793/180.0);
        for (int i=0; i<4; ++i)
            for (int j=0; j<i; ++j) {
                double dot=0;
                for (int k=0; k<3; ++k) dot+=normals[i][k]*normals[j][k];
                if (dot+1e-12<limit) return false;
            }
        return true;
    };

    for (size_t i=0; i<candidates.size(); ++i) {
        const Candidate& c=candidates[i];
        if (!normal_compatible(c)) continue;
        const double max_warpage=options.face_warpage.empty()
            ? options.maximum_warpage
            : std::min({options.maximum_warpage,options.face_warpage[c.f0],options.face_warpage[c.f1]});
        if (warpage(c)>max_warpage) continue;
        if (options.forbidden_pairs.count({std::min(c.f0,c.f1),std::max(c.f0,c.f1)})) continue;
        result.best_quality[c.f0]=std::max(result.best_quality[c.f0],c.quality);
        result.best_quality[c.f1]=std::max(result.best_quality[c.f1],c.quality);
        if (!std::isfinite(c.quality) || c.quality<options.minimum_scaled_jacobian) continue;
        if (corner_score(c)+1e-12<std::sin(options.minimum_corner_degrees*3.141592653589793/180.0)) continue;
        ++result.eligible_degree[c.f0]; ++result.eligible_degree[c.f1];
        const auto edge=graph.addEdge(nodes[c.f0],nodes[c.f1]);
        weight[edge]=priority+face_bonus*(options.preferred_faces.count(c.f0)+
            options.preferred_faces.count(c.f1))+
            std::llround((options.corner_objective
                ? std::min(1.0,corner_score(c)/std::sin(5.0*3.141592653589793/180.0))
                : std::clamp(c.quality,0.0,1.0))*1000);
        index[edge]=i;
        ++result.eligible_count;
    }

    std::vector<size_t> selected;
    if (options.weighted) {
        lemon::MaxWeightedMatching<lemon::SmartGraph, decltype(weight)> solver(graph,weight);
        solver.run();
        for (lemon::SmartGraph::EdgeIt edge(graph); edge!=lemon::INVALID; ++edge)
            if (solver.matching(edge)) selected.push_back(index[edge]);
    } else {
        lemon::MaxMatching<lemon::SmartGraph> solver(graph);
        solver.run();
        for (lemon::SmartGraph::EdgeIt edge(graph); edge!=lemon::INVALID; ++edge)
            if (solver.matching(edge)) selected.push_back(index[edge]);
    }
    std::vector<bool> covered(triangles.size(),false);
    for (size_t i:selected) {
        const Candidate& c=candidates[i];
        if (covered[c.f0] || covered[c.f1]) throw std::runtime_error("overlapping matched pairs");
        covered[c.f0]=covered[c.f1]=true;
        result.quads.push_back(c.q);
        result.quad_source_faces.push_back({c.f1,c.f0});
        result.quad_quality.push_back(c.quality);
    }
    for (size_t i=0; i<triangles.size(); ++i)
        if (!covered[i]) result.unmatched_faces.push_back(i);
    return result;
}

}  // namespace trellis
