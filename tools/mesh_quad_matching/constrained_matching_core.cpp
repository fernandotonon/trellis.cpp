#include "quad_matching.h"
#ifdef TRELLIS_HAVE_QUAD_COLLISION
#include "quad_collision.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    if (argc < 5 || argc > 7) return 2;
    try {
        trellis::QuadMatchOptions options;
        options.minimum_scaled_jacobian = std::stod(argv[3]);
        if (std::string(argv[4]) == "weighted") options.weighted = true;
        else if (std::string(argv[4]) == "cardinality") options.weighted = false;
        else return 2;
        if (argc == 7) options.maximum_warpage = std::stod(argv[6]);
        if (argc >= 6) {
            std::ifstream bans(argv[5]);
            if (!bans) return 2;
            size_t a,b;
            while (bans >> a >> b) options.forbidden_pairs.insert({std::min(a,b),std::max(a,b)});
            if (!bans.eof()) return 2;
        }
        if (const char* value = std::getenv("TRELLIS_PROBE_MIN_CORNER_DEGREES"))
            options.minimum_corner_degrees = std::stod(value);
        if (const char* value = std::getenv("TRELLIS_PROBE_MAX_NORMAL_SPAN"))
            options.maximum_normal_span_degrees = std::stod(value);
        options.corner_objective = std::getenv("TRELLIS_PROBE_MATCH_CORNERS") != nullptr;
        if (const char* path = std::getenv("TRELLIS_PROBE_MATCH_PREFERRED_FACES")) {
            std::ifstream preferred(path);
            if (!preferred) return 2;
            size_t face;
            while (preferred >> face)
                if (!options.preferred_faces.insert(face).second) return 2;
            if (!preferred.eof()) return 2;
        }
        std::ifstream input(argv[1]);
        if (!input) return 2;
        std::vector<std::array<float,3>> vertices;
        std::vector<std::array<uint32_t,3>> triangles;
        std::string line;
        while (std::getline(input,line)) {
            std::istringstream stream(line);
            std::string tag;
            stream >> tag;
            if (tag == "v") {
                std::array<float,3> point;
                if (!(stream >> point[0] >> point[1] >> point[2])) return 2;
                vertices.push_back(point);
            } else if (tag == "f") {
                std::array<uint32_t,3> face;
                for (auto& index:face) {
                    std::string token;
                    if (!(stream >> token)) return 2;
                    const size_t parsed=std::stoull(token.substr(0,token.find('/')));
                    if (!parsed || parsed>vertices.size()) return 2;
                    index=uint32_t(parsed-1);
                }
                std::string extra;
                if (stream >> extra) return 2;
                triangles.push_back(face);
            }
        }
        if (const char* path = std::getenv("TRELLIS_PROBE_FACE_WARPAGE")) {
            std::ifstream limits(path);
            if (!limits) return 2;
            options.face_warpage.assign(triangles.size(),options.maximum_warpage);
            std::set<size_t> seen;
            size_t face;
            double limit;
            while (limits >> face) {
                if (!(limits >> limit) || face>=triangles.size() || !std::isfinite(limit) ||
                    limit<0 || !seen.insert(face).second) return 2;
                options.face_warpage[face]=std::min(options.maximum_warpage,limit);
            }
            if (!limits.eof()) return 2;
        }
        trellis::QuadMatchResult result;
        if (std::getenv("TRELLIS_PROBE_COLLISION_SAFE")) {
#ifdef TRELLIS_HAVE_QUAD_COLLISION
            auto safe=trellis::match_triangles_to_quads_collision_safe(vertices,triangles,options);
            for (size_t i=0; i<safe.rounds.size(); ++i) {
                const auto& round=safe.rounds[i];
                std::cout << "COLLISION round=" << i << " crossings="
                          << round.collision.cross_polygon_intersections << " degeneracies="
                          << round.collision.degenerate_triangles << " duplicates="
                          << round.collision.duplicate_triangle_pairs << " chords="
                          << round.collision.diagonal_edge_conflicts << " new_forbidden="
                          << round.new_forbidden_pairs << " total_forbidden="
                          << round.forbidden_pairs << std::endl;
            }
            result=std::move(safe.match);
#else
            throw std::runtime_error("collision-safe matching is disabled in this build");
#endif
        } else result=trellis::match_triangles_to_quads(vertices,triangles,options);
        std::cout << "GRAPH triangles=" << triangles.size() << " candidates="
                  << result.candidate_count << " eligible=" << result.eligible_count << std::endl;
        std::ofstream out(argv[2]), provenance(std::string(argv[2])+".source-ids"),
                      parents(std::string(argv[2])+".source-faces"),
                      unmatched(std::string(argv[2])+".unmatched.tsv");
        if (!out || !provenance || !parents || !unmatched) return 2;
        out << std::setprecision(9);
        for (size_t i=0; i<vertices.size(); ++i) {
            const auto& p=vertices[i];
            out << "v " << p[0] << ' ' << p[1] << ' ' << p[2] << '\n';
            provenance << i << '\n';
        }
        for (size_t i=0; i<result.quads.size(); ++i) {
            out << "f";
            for (uint32_t index:result.quads[i]) out << ' ' << index+1;
            out << '\n';
            parents << result.quad_source_faces[i][0] << ' '
                    << result.quad_source_faces[i][1] << '\n';
        }
        for (size_t face:result.unmatched_faces) {
            out << "f";
            for (uint32_t index:triangles[face]) out << ' ' << index+1;
            out << '\n';
            parents << face << '\n';
            unmatched << face << '\t' << result.eligible_degree[face] << '\t'
                      << result.best_quality[face] << '\n';
        }
        auto quality=result.quad_quality;
        std::sort(quality.begin(),quality.end());
        std::cout << "RESULT quads=" << result.quads.size()
                  << " triangles=" << result.unmatched_faces.size()
                  << " min_scaled_jacobian=" << (quality.empty()?0:quality[0])
                  << " q01=" << (quality.empty()?0:quality[quality.size()/100])
                  << " median_quality=" << (quality.empty()?0:quality[quality.size()/2])
                  << std::endl;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
