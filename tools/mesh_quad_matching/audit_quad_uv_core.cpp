#include "quad_uv_repair.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 4) return 2;
    try {
        const auto quads = read_uv_quads(argv[1]);
        std::ifstream input(argv[2],std::ios::binary);
        uint32_t counts[2]{};
        input.read(reinterpret_cast<char*>(counts),sizeof(counts));
        if (!input || !counts[0] || !counts[1] ||
            std::filesystem::file_size(argv[2]) != 8+uint64_t(counts[0])*20+uint64_t(counts[1])*12)
            throw std::runtime_error("Invalid UV dump");
        std::vector<UvPoint3> positions(counts[0]);
        std::vector<std::array<float,2>> uv(counts[0]);
        std::vector<std::array<uint32_t,3>> faces(counts[1]);
        input.read(reinterpret_cast<char*>(positions.data()),positions.size()*12);
        input.read(reinterpret_cast<char*>(uv.data()),uv.size()*8);
        input.read(reinterpret_cast<char*>(faces.data()),faces.size()*12);
        if (!input) throw std::runtime_error("Truncated UV dump");
        for (const auto& p:positions)
            for (float x:p) if (!std::isfinite(x)) throw std::runtime_error("Nonfinite UV position");
        for (const auto& p:uv)
            for (float x:p) if (!std::isfinite(x)) throw std::runtime_error("Nonfinite UV coordinate");
        for (const auto& f:faces)
            for (uint32_t v:f) if (v>=positions.size()) throw std::runtime_error("Invalid UV index");
        const auto mapping=match_quad_uv_faces(quads,positions,faces);
        const std::string prefix=argv[3];
        std::ofstream incompatible(prefix+".incompatible-quads.txt");
        std::ofstream nonconvex(prefix+".nonconvex-faces.txt");
        std::ofstream repair(prefix+".repair-faces.txt");
        if (!incompatible || !nonconvex || !repair)
            throw std::runtime_error("Cannot write quad UV audit sidecars");
        size_t seams=0,bad_convex=0;
        for (size_t qi=0; qi<quads.q.size(); ++qi) {
            const auto& q=quads.q[qi];
            const auto ids=mapping[qi];
            const auto& first=faces[ids[0]];
            const auto& second=faces[ids[1]];
            const auto at=[&](const std::array<uint32_t,3>& face, uint32_t vertex) {
                return uv[uv_vertex_at(positions,face,quads.p[vertex])];
            };
            const std::array<std::array<float,2>,4> corners{
                at(first,q[0]),at(first,q[1]),at(first,q[2]),at(second,q[3])};
            const bool seam=at(first,q[0])!=at(second,q[0]) ||
                            at(first,q[2])!=at(second,q[2]);
            double turns[4]{};
            for (int i=0;i<4;++i) {
                const auto& a=corners[i];
                const auto& b=corners[(i+1)%4];
                const auto& c=corners[(i+2)%4];
                const double ex=double(b[0])-a[0],ey=double(b[1])-a[1];
                const double fx=double(c[0])-b[0],fy=double(c[1])-b[1];
                turns[i]=ex*fy-ey*fx;
            }
            bool positive=true,negative=true;
            for (double turn:turns) { positive &= turn>0; negative &= turn<0; }
            const bool concave=!(positive || negative);
            seams+=seam;bad_convex+=concave;
            if (seam) incompatible << qi << '\n';
            if (concave) nonconvex << ids[0] << '\n';
            if (seam || concave) repair << ids[0] << '\n';
        }
        std::ofstream report(prefix);
        report << "{\"quads\":" << quads.q.size()
               << ",\"geometry_triangles_matched\":" << faces.size()
               << ",\"quads_with_internal_uv_seam\":" << seams
               << ",\"nonconvex_quad_uvs\":" << bad_convex
               << ",\"all_quad_uvs_strictly_convex\":" << (bad_convex?"false":"true")
               << ",\"four_corner_uv_export_possible\":" << ((seams||bad_convex)?"false":"true")
               << "}\n";
        incompatible.close();nonconvex.close();repair.close();report.close();
        if (!incompatible || !nonconvex || !repair || !report)
            throw std::runtime_error("Cannot finish quad UV audit");
        std::cout << "quads=" << quads.q.size() << " seams=" << seams
                  << " nonconvex=" << bad_convex << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
