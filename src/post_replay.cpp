// post-replay — re-run the post-neural stages (weld, hole fill, decimation, UV
// bake, GLB write) from a TRELLIS_DUMP_POST dump, skipping the ~10-minute
// neural pipeline. Development harness for iterating on mesh/texture
// post-processing.
//
//   post-replay <dump.bin|triangles.obj> <out.glb> [--box-uv] [--faces N] [--atlas T]
//               [--decim GRID] [--no-weld] [--no-fill]
//               [--keep-components] [--dump-geometry mesh.obj]
//               [--quad-match-output mesh.obj | --quad-safe-output mesh.obj]
//               [--quad-refine-output quads.obj] [--quad-refine-forced ids.txt]
//               [--quad-refine-seed edges.tsv] [--quad-refine-qualified]
//               [--quad-repair-source original.post]
//               [--quad-source-cover-output selected.obj]
#include "uv_bake.h"
#include "tri_bvh.h"
#include "remesh_dc.h"
#include "mesh_glb.h"
#include "mesh_shape.h"
#include "quad_refinement.h"
#include "quad_refinement_io.h"
#include "quad_components.h"
#include "quad_topology.h"
#ifdef TRELLIS_HAVE_QUAD_MATCHING
#include "quad_matching.h"
#ifdef TRELLIS_HAVE_QUAD_COLLISION
#include "quad_collision.h"
#include "quad_local_repair.h"
#include "quad_quality.h"
#endif
#endif
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <exception>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using trellis::VoxelPbr;

static double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// boundary/non-manifold audit over the welded index space (positions assumed welded)
#include <unordered_map>
static void audit(const char* tag, const std::vector<int32_t>& faces) {
    std::unordered_map<uint64_t,int> e;
    e.reserve(faces.size() * 2);
    const size_t F = faces.size() / 3;
    auto k = [](int a, int b){ if (a>b){int t=a;a=b;b=t;} return ((uint64_t)(uint32_t)a<<32)|(uint32_t)b; };
    for (size_t f = 0; f < F; ++f)
        for (int j = 0; j < 3; ++j) e[k(faces[3*f+j], faces[3*f+(j+1)%3])]++;
    size_t nb = 0, nm = 0;
    for (auto& kv : e) { if (kv.second == 1) ++nb; else if (kv.second > 2) ++nm; }
    printf("  [audit] %-22s F=%-9zu boundary_edges=%-7zu nonmanifold=%zu\n", tag, F, nb, nm);
    fflush(stdout);
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: post-replay <dump.bin|triangles.obj> <out.glb> [opts] [--dump-geometry mesh.obj]\n"); return 1; }
    const char* dump = argv[1];
    const char* out = argv[2];
    const char* geometry_dump = nullptr;
    const char* geometry_post_dump = nullptr;
    const char* quad_match_dump = nullptr;
    const char* quad_refine_dump = nullptr;
    const char* quad_refine_forced = nullptr;
    const char* quad_refine_seed = nullptr;
    const char* quad_repair_source = nullptr;
    const char* quad_source_cover_dump = nullptr;
    bool quad_refine_qualified = false;
    bool quad_match_safe = false;
    bool boxuv = false, do_weld = true, do_fill = true, do_bake = true, do_remesh = true, do_snap = true;
    bool keep_components = false, shape_flips = false, shape_smooth = false;
    float shape_min_angle = 1.0f;
    float shape_max_move = 0.001f;
    int band = 1;
    int faces_target = 300000, atlas = 2048, decim = -1, geometry_res = 0;
    for (int i = 3; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--box-uv") boxuv = true;
        else if (a == "--faces" && i+1 < argc) faces_target = atoi(argv[++i]);
        else if (a == "--atlas" && i+1 < argc) atlas = atoi(argv[++i]);
        else if (a == "--decim" && i+1 < argc) decim = atoi(argv[++i]);
        else if (a == "--no-weld") do_weld = false;
        else if (a == "--no-fill") do_fill = false;
        else if (a == "--no-bake") do_bake = false;
        else if (a == "--no-remesh") do_remesh = false;
        else if (a == "--band" && i+1 < argc) band = atoi(argv[++i]);
        else if (a == "--geometry-res" && i+1 < argc) {
            char* end = nullptr;
            const char* value = argv[++i];
            const long parsed = strtol(value, &end, 10);
            if (end == value || *end || (parsed != 512 && parsed != 1024 && parsed != 1536)) {
                fprintf(stderr, "invalid --geometry-res\n");
                return 1;
            }
            geometry_res = int(parsed);
        }
        else if (a == "--no-snap") do_snap = false;
        else if (a == "--keep-components") keep_components = true;
        else if (a == "--shape-flips") shape_flips = true;
        else if (a == "--shape-smooth") shape_smooth = true;
        else if (a == "--shape-min-angle" && i+1 < argc) {
            char* end = nullptr;
            shape_min_angle = strtof(argv[++i], &end);
            if (end == argv[i] || *end || !std::isfinite(shape_min_angle) ||
                shape_min_angle <= 0.0f || shape_min_angle > 30.0f) {
                fprintf(stderr, "invalid --shape-min-angle\n");
                return 1;
            }
        }
        else if (a == "--shape-max-move" && i+1 < argc) {
            char* end = nullptr;
            shape_max_move = strtof(argv[++i], &end);
            if (end == argv[i] || *end || !std::isfinite(shape_max_move) ||
                shape_max_move <= 0.0f || shape_max_move > 0.01f) {
                fprintf(stderr, "invalid --shape-max-move\n");
                return 1;
            }
        }
        else if (a == "--dump-geometry" && i+1 < argc) geometry_dump = argv[++i];
        else if (a == "--dump-geometry-post" && i+1 < argc) geometry_post_dump = argv[++i];
        else if (a == "--quad-match-output" && i+1 < argc) {
            if (quad_match_dump) return 2;
            quad_match_dump = argv[++i];
        }
        else if (a == "--quad-safe-output" && i+1 < argc) {
            if (quad_match_dump) return 2;
            quad_match_dump = argv[++i];
            quad_match_safe = true;
        }
        else if (a == "--quad-refine-output" && i+1 < argc) quad_refine_dump = argv[++i];
        else if (a == "--quad-refine-forced" && i+1 < argc) quad_refine_forced = argv[++i];
        else if (a == "--quad-refine-seed" && i+1 < argc) quad_refine_seed = argv[++i];
        else if (a == "--quad-refine-qualified") quad_refine_qualified = true;
        else if (a == "--quad-repair-source" && i+1 < argc) quad_repair_source = argv[++i];
        else if (a == "--quad-source-cover-output" && i+1 < argc) quad_source_cover_dump = argv[++i];
        else { fprintf(stderr, "unknown or incomplete option: %s\n", a.c_str()); return 1; }
    }
    if ((quad_refine_dump && !quad_match_dump) ||
        ((quad_refine_forced || quad_refine_seed || quad_refine_qualified) && !quad_refine_dump)) {
        fprintf(stderr, "quad refinement requires matching output and refinement output paths\n");
        return 1;
    }
    if (quad_repair_source && (!quad_match_safe || !quad_refine_qualified || !quad_refine_dump)) {
        fprintf(stderr, "--quad-repair-source requires qualified collision-safe matching and refinement\n");
        return 1;
    }
    if (quad_source_cover_dump && (!quad_repair_source || !quad_refine_qualified)) {
        fprintf(stderr, "--quad-source-cover-output requires qualified quad refinement and --quad-repair-source\n");
        return 1;
    }

    int V = 0, F = 0, Mv = 0, res = 0;
    std::vector<float> verts, pbr6;
    std::vector<int32_t> faces;
    std::vector<std::array<int,3>> coords;
    const std::string input_path = dump;
    if (input_path.size() >= 4 && input_path.substr(input_path.size()-4) == ".obj") {
        if (!geometry_res || do_bake) {
            fprintf(stderr, "triangle OBJ input requires --geometry-res and --no-bake\n");
            return 1;
        }
        std::ifstream input(dump);
        if (!input) { fprintf(stderr, "cannot open %s\n", dump); return 1; }
        std::string line;
        while (std::getline(input, line)) {
            std::istringstream stream(line);
            std::string tag;
            stream >> tag;
            if (tag == "v") {
                float x, y, z;
                std::string extra;
                if (!(stream >> x >> y >> z) || (stream >> extra) ||
                    !std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))
                    return 1;
                verts.insert(verts.end(), {x, y, z});
            } else if (tag == "f") {
                int a, b, c;
                std::string extra;
                if (!(stream >> a >> b >> c) || (stream >> extra) ||
                    a <= 0 || b <= 0 || c <= 0 ||
                    a > int(verts.size()/3) || b > int(verts.size()/3) ||
                    c > int(verts.size()/3)) return 1;
                faces.insert(faces.end(), {a-1, b-1, c-1});
            } else if (!tag.empty() && tag[0] != '#') {
                fprintf(stderr, "unsupported OBJ record: %s\n", tag.c_str());
                return 1;
            }
        }
        if (verts.empty() || faces.empty() || !input.eof()) return 1;
        V = int(verts.size()/3); F = int(faces.size()/3); res = geometry_res;
    } else {
        FILE* f = fopen(dump, "rb");
        if (!f) { fprintf(stderr, "cannot open %s\n", dump); return 1; }
        if (fread(&V,4,1,f)+fread(&F,4,1,f)+fread(&Mv,4,1,f)+fread(&res,4,1,f) != 4 ||
            V <= 0 || F <= 0 || Mv < 0 || res <= 0) return 1;
        if (!geometry_res) geometry_res = res;
        verts.resize((size_t)V*3);
        faces.resize((size_t)F*3);
        coords.resize((size_t)Mv);
        pbr6.resize((size_t)Mv*6);
        if (fread(verts.data(),4,verts.size(),f) != verts.size()) return 1;
        if (fread(faces.data(),4,faces.size(),f) != faces.size()) return 1;
        for (auto& c : coords) if (fread(c.data(),4,3,f) != 3) return 1;
        if (fread(pbr6.data(),4,pbr6.size(),f) != pbr6.size()) return 1;
        fclose(f);
    }
    printf("loaded: V=%d F=%d voxels=%d pbr_res=%d geometry_res=%d\n", V, F, Mv, res, geometry_res);

    double t = now();
    if (do_weld) trellis::weld_vertices(verts, faces, nullptr, 1.0f / ((float)geometry_res * 8.0f));
    printf("  [weld %.1fs]\n", now()-t); t = now();
    audit("weld", faces);
    if (do_fill) trellis::fill_small_holes(faces);
    printf("  [fill %.1fs]\n", now()-t); t = now();
    audit("fill_small_holes", faces);

    trellis::TriBvh bvh = trellis::TriBvh::build(verts.data(), (int64_t)verts.size()/3,
                                                 faces.data(), (int64_t)faces.size()/3);
    printf("  [bvh %.1fs]\n", now()-t); t = now();
    trellis::Mesh rm;
    if (do_remesh) {
        rm = trellis::remesh_narrow_band_dc(verts.data(), (int64_t)verts.size()/3,
                                            faces.data(), (int64_t)faces.size()/3, bvh, geometry_res, band);
        printf("  [remesh %.1fs]\n", now()-t); t = now();
        audit("remesh", rm.faces);
        // match the CLI: clean degenerates/unify winding, drop floater components
        if (rm.F() > 0) {
            trellis::clean_mesh(rm.V(), rm.faces);
            audit("clean_mesh", rm.faces);
            if (!keep_components) {
                int ndrop = trellis::drop_small_components(rm.verts, rm.faces, 0.02f);
                printf("  [clean+drop %.1fs] dropped=%d\n", now()-t, ndrop); t = now();
                audit("drop_components", rm.faces);
            }
        }
    }
    const std::vector<float>& sverts = rm.F() > 0 ? rm.verts : verts;
    const std::vector<int32_t>& sfaces = rm.F() > 0 ? rm.faces : faces;

    std::vector<float> dv, dp; std::vector<int32_t> df;
    if (decim > 0) trellis::decimate_cluster(sverts, (int)sverts.size()/3, sfaces, (int)sfaces.size()/3, {}, decim, dv, df, dp);
    else if (decim == 0) { dv = sverts; df = sfaces; }
    else {
        // match the CLI: faithful QEM port (not the old meshopt/FQMS decimate_simplify)
        trellis::decimate_qem(sverts, (int)sverts.size()/3, sfaces, (int)sfaces.size()/3, faces_target, dv, df);
        audit("decimate_qem", df);
        if (!std::getenv("TRELLIS_QEM_PRESERVE_TOPOLOGY") &&
            !std::getenv("TRELLIS_QEM_COLLISION_GUARD")) {
            if (do_weld) {
                trellis::weld_vertices(dv, df, nullptr, 1.0f / ((float)geometry_res * 8.0f));
                audit("weld2", df);
            }
            if (do_fill) {
                trellis::fill_small_holes(df);
                audit("fill2", df);
            }
        }
        if (!keep_components) {
            int ndrop2 = trellis::drop_small_components(dv, df, 0.03f);
            if (ndrop2) printf("  dropped %d more comps\n", ndrop2);
            audit("drop2", df);
        }
    }
    printf("  [decimate %.1fs]\n", now()-t); t = now();
    if (shape_flips) {
        auto result = trellis::flip_skinny_triangles(dv,df,shape_min_angle);
        printf("  shape_flips: accepted=%d collisions=%d swept=%d\n",
               result.flips,result.collision_rejections,result.swept_rejections);
        audit("shape_flips",df);
    }
    if (shape_smooth) {
        auto result = trellis::smooth_skinny_triangles(dv,df,shape_min_angle,shape_max_move);
        printf("  shape_smooth: moved=%d collisions=%d max_displacement=%.9g\n",
               result.moved,result.collision_rejections,result.maximum_displacement);
        audit("shape_smooth",df);
    }
    if (quad_match_dump) {
#ifdef TRELLIS_HAVE_QUAD_MATCHING
        std::vector<std::array<float,3>> positions(dv.size()/3);
        std::vector<std::array<uint32_t,3>> triangles(df.size()/3);
        for (size_t i=0; i<positions.size(); ++i)
            positions[i] = {dv[3*i], dv[3*i+1], dv[3*i+2]};
        for (size_t i=0; i<triangles.size(); ++i)
            triangles[i] = {uint32_t(df[3*i]), uint32_t(df[3*i+1]), uint32_t(df[3*i+2])};
        trellis::QuadMatchOptions options;
        options.maximum_normal_span_degrees = 90;
        trellis::QuadMatchResult result;
        if (quad_match_safe) {
#ifdef TRELLIS_HAVE_QUAD_COLLISION
            trellis::SafeQuadMatchResult safe;
            try {
                safe = trellis::match_triangles_to_quads_collision_safe(positions, triangles, options);
            } catch (const std::exception& error) {
                fprintf(stderr,"quad collision-safe matching failed: %s\n",error.what());
                return 1;
            }
            for (size_t i=0; i<safe.rounds.size(); ++i) {
                const auto& round = safe.rounds[i];
                printf("  quad collision round %zu: crossings=%zu degeneracies=%zu duplicates=%zu chords=%zu new_bans=%zu\n",
                       i, round.collision.cross_polygon_intersections,
                       round.collision.degenerate_triangles,
                       round.collision.duplicate_triangle_pairs,
                       round.collision.diagonal_edge_conflicts,
                       round.new_forbidden_pairs);
            }
            result = std::move(safe.match);
#else
            fprintf(stderr, "--quad-safe-output requires TRELLIS_RETOPO_COLLISION=ON\n");
            return 1;
#endif
        } else result = trellis::match_triangles_to_quads(positions, triangles, options);
        std::ofstream matched(quad_match_dump);
        std::ofstream parents(std::string(quad_match_dump) + ".source-faces");
        std::ofstream ids(std::string(quad_match_dump) + ".source-ids");
        if (!matched || !parents || !ids) {
            fprintf(stderr, "cannot write quad match output %s\n", quad_match_dump);
            return 1;
        }
        matched << std::setprecision(9);
        for (size_t i=0; i<positions.size(); ++i) {
            const auto& p=positions[i];
            matched << "v " << p[0] << ' ' << p[1] << ' ' << p[2] << '\n';
            ids << i << '\n';
        }
        for (size_t i=0; i<result.quads.size(); ++i) {
            matched << "f";
            for (uint32_t index:result.quads[i]) matched << ' ' << index+1;
            matched << '\n';
            parents << result.quad_source_faces[i][0] << ' '
                    << result.quad_source_faces[i][1] << '\n';
        }
        for (size_t face:result.unmatched_faces) {
            matched << "f";
            for (uint32_t index:triangles[face]) matched << ' ' << index+1;
            matched << '\n';
            parents << face << '\n';
        }
        if (!matched || !parents || !ids) return 1;
        printf("  quad match: candidates=%zu eligible=%zu quads=%zu residual_triangles=%zu -> %s\n",
               result.candidate_count, result.eligible_count, result.quads.size(),
               result.unmatched_faces.size(), quad_match_dump);
        if (quad_refine_dump) {
            std::vector<std::vector<uint32_t>> mixed;
            mixed.reserve(result.quads.size()+result.unmatched_faces.size());
            for (const auto& quad:result.quads)
                mixed.push_back({quad[0],quad[1],quad[2],quad[3]});
            for (size_t face:result.unmatched_faces)
                mixed.push_back({triangles[face][0],triangles[face][1],triangles[face][2]});
            trellis::QuadRefinementOptions refine_options;
            refine_options.surface = true;
            refine_options.center_normal_span_degrees = 90;
            if (quad_refine_forced) {
                std::ifstream forced(quad_refine_forced);
                if (!forced) return 1;
                size_t id;
                while (forced >> id) refine_options.forced_faces.insert(id);
                if (!forced.eof()) return 1;
            }
            if (quad_refine_seed) {
                refine_options.use_split_seed = true;
                std::ifstream seed(quad_refine_seed);
                if (!seed) return 1;
                size_t a, b;
                while (seed >> a >> b)
                    refine_options.split_seed.insert({std::min(a,b),std::max(a,b)});
                if (!seed.eof()) return 1;
            }
            trellis::QuadRefinementResult refined;
            if (quad_refine_qualified) {
                trellis::QualifiedQuadRefinementResult qualified;
                try {
                    qualified = trellis::refine_quads_until_qualified(
                        positions,mixed,refine_options,35,quad_repair_source != nullptr);
                } catch (const std::exception& error) {
                    fprintf(stderr, "quad refinement failed: %s\n", error.what());
                    return 1;
                }
                for (size_t i=0; i<qualified.rounds.size(); ++i) {
                    const auto& round = qualified.rounds[i];
                    printf("  quad refinement round %zu: forced=%zu new=%zu quality=%zu shape=%zu unproven=%zu quads=%zu\n",
                           i,round.forced_parents,round.new_forced_parents,
                           round.bad_quality_parents,round.bad_shape_parents,
                           round.unproven_parents,round.quad_faces);
                }
                refined = std::move(qualified.mesh);
            } else refined = trellis::refine_quads(positions,mixed,refine_options);
#ifdef TRELLIS_HAVE_QUAD_COLLISION
            std::vector<std::array<float,3>> source_vertices;
            std::vector<std::array<int32_t,3>> source_faces;
            if (quad_repair_source) {
                try {
                    std::ifstream source(quad_repair_source,std::ios::binary);
                    int32_t header[4]{};
                    source.read(reinterpret_cast<char*>(header),sizeof(header));
                    if (!source || header[0] <= 0 || header[1] <= 0)
                        throw std::runtime_error("Invalid quad repair source POST");
                    source_vertices.resize(header[0]);
                    source_faces.resize(header[1]);
                    source.read(reinterpret_cast<char*>(source_vertices.data()),
                                source_vertices.size()*sizeof(source_vertices[0]));
                    source.read(reinterpret_cast<char*>(source_faces.data()),
                                source_faces.size()*sizeof(source_faces[0]));
                    if (!source) throw std::runtime_error("Truncated quad repair source POST");
                    const auto shape = trellis::repair_quad_shapes(refined.vertices,refined.quads);
                    printf("  quad shape repair: bad_before=%zu bad_after=%zu moved=%zu\n",
                           shape.bad_quads_before,shape.bad_quads_after,shape.moves.size());
                    const auto tree = trellis::TriBvh::build(
                        reinterpret_cast<const float*>(source_vertices.data()),source_vertices.size(),
                        reinterpret_cast<const int32_t*>(source_faces.data()),source_faces.size());
                    const auto repair = trellis::repair_quad_collisions(
                        refined.vertices,refined.quads,
                        [&](const std::array<float,3>& point) {
                            const auto hit = tree.closest(point.data());
                            if (hit.face < 0) throw std::runtime_error("Missing quad repair source projection");
                            return std::array<float,3>{hit.point[0],hit.point[1],hit.point[2]};
                        });
                    printf("  quad local repair: crossings_before=%zu crossings_after=%zu moved=%zu\n",
                           repair.before.cross_polygon_intersections,
                           repair.after.cross_polygon_intersections,repair.moves.size());
                    for (size_t qi=0; qi<refined.quads.size(); ++qi) {
                        std::array<trellis::QuadPoint,4> points;
                        for (int j=0; j<4; ++j) {
                            const auto& p = refined.vertices[refined.quads[qi][j]];
                            points[j] = {p[0],p[1],p[2]};
                        }
                        refined.quality[qi] = trellis::quad_min_scaled_jacobian(points);
                    }
                    const auto gate = trellis::audit_quad_refinement(mixed,refined);
                    if (!gate.bad_quality_parents.empty() || !gate.bad_shape_parents.empty() ||
                        !gate.unproven_parents.empty() || gate.nonpositive_witness_triangles)
                        throw std::runtime_error("Quad repair left unresolved refinement gates");
                } catch (const std::exception& error) {
                    fprintf(stderr, "quad repair failed: %s\n",error.what());
                    return 1;
                }
            }
            const auto collision = trellis::audit_quad_collisions(refined.vertices, refined.quads);
            printf("  quad refined collision: crossings=%zu degeneracies=%zu duplicates=%zu chords=%zu affected=%zu\n",
                   collision.cross_polygon_intersections, collision.degenerate_triangles,
                   collision.duplicate_triangle_pairs, collision.diagonal_edge_conflicts,
                   collision.affected_polygons.size());
            if (quad_refine_qualified && !collision.clean()) {
                fprintf(stderr, "quad refinement has unresolved collisions\n");
                return 1;
            }
#else
            if (quad_repair_source) {
                fprintf(stderr, "--quad-repair-source requires TRELLIS_RETOPO_COLLISION=ON\n");
                return 1;
            }
#endif
            std::vector<size_t> face_sizes;
            face_sizes.reserve(mixed.size());
            for (const auto& face:mixed) face_sizes.push_back(face.size());
            trellis::write_quad_refinement_artifacts(quad_refine_dump,refined,face_sizes);
#ifdef TRELLIS_HAVE_QUAD_COLLISION
            if (quad_source_cover_dump) {
                try {
                    const auto labels=trellis::label_quad_components(refined.vertices.size(),refined.quads);
                    const auto cover=trellis::select_source_covering_quad_components(
                        source_vertices,source_faces,refined.vertices,refined.quads,labels,.008,12);
                    const auto filtered=trellis::filter_quad_components(
                        refined.vertices,refined.quads,labels,cover.keep);
                    const auto topology=trellis::audit_quad_topology(
                        filtered.vertices.size(),filtered.quads);
                    if (!topology.clean())
                        throw std::runtime_error("Source-covered quads fail topology audit");
                    printf("  quad topology: components=%zu boundary=%zu nonmanifold_edges=%zu nonmanifold_vertices=%zu winding=%zu duplicates=%zu\n",
                           topology.components,topology.boundary_edges,
                           topology.nonmanifold_edges,topology.nonmanifold_vertices,
                           topology.winding_conflicts,topology.duplicate_faces);
                    std::ofstream output(quad_source_cover_dump);
                    if (!output) throw std::runtime_error("Cannot write source-covered quad OBJ");
                    output << std::setprecision(9);
                    for (const auto& p:filtered.vertices)
                        output << "v " << p[0] << ' ' << p[1] << ' ' << p[2] << '\n';
                    for (const auto& q:filtered.quads)
                        output << "f " << q[0]+1 << ' ' << q[1]+1 << ' '
                               << q[2]+1 << ' ' << q[3]+1 << '\n';
                    output.close();
                    if (!output) throw std::runtime_error("Cannot finish source-covered quad OBJ");
                    printf("  quad source cover: rounds=%zu selected_components=%zu quads=%zu upper_bound=%.9g reverse_bound=%.9g queries=%llu reverse_queries=%llu -> %s\n",
                           cover.rounds,cover.keep.size(),filtered.quads.size(),
                           cover.maximum_leaf_upper_bound,cover.maximum_reverse_leaf_upper_bound,
                           static_cast<unsigned long long>(cover.distance_queries),
                           static_cast<unsigned long long>(cover.reverse_distance_queries),quad_source_cover_dump);
                } catch (const std::exception& error) {
                    fprintf(stderr,"quad source cover failed: %s\n",error.what());
                    return 1;
                }
            }
#endif
        }
#else
        fprintf(stderr, "--quad-match-output requires TRELLIS_RETOPO_MATCHING=ON\n");
        return 1;
#endif
    }
    if (geometry_dump) {
        FILE* mesh = fopen(geometry_dump, "wb");
        if (!mesh) { fprintf(stderr, "cannot write %s\n", geometry_dump); return 1; }
        bool ok = true;
        for (size_t i = 0; i + 2 < dv.size() && ok; i += 3)
            ok = fprintf(mesh, "v %.9g %.9g %.9g\n", dv[i], dv[i+1], dv[i+2]) > 0;
        for (size_t i = 0; i + 2 < df.size() && ok; i += 3)
            ok = fprintf(mesh, "f %d %d %d\n", df[i]+1, df[i+1]+1, df[i+2]+1) > 0;
        if (fclose(mesh) != 0) ok = false;
        if (!ok) { fprintf(stderr, "failed to write %s\n", geometry_dump); return 1; }
        printf("wrote geometry %s (V=%zu F=%zu)\n", geometry_dump, dv.size()/3, df.size()/3);
    }
    if (geometry_post_dump) {
        FILE* post = fopen(geometry_post_dump, "wb");
        if (!post) { fprintf(stderr, "cannot write %s\n", geometry_post_dump); return 1; }
        const int32_t header[4] = {int32_t(dv.size()/3), int32_t(df.size()/3), 0,
                                   int32_t(geometry_res)};
        bool ok = fwrite(header, sizeof(header), 1, post) == 1 &&
                  fwrite(dv.data(), sizeof(float), dv.size(), post) == dv.size() &&
                  fwrite(df.data(), sizeof(int32_t), df.size(), post) == df.size();
        ok = fclose(post) == 0 && ok;
        if (!ok) { fprintf(stderr, "failed to write %s\n", geometry_post_dump); return 1; }
        printf("wrote geometry POST %s (V=%zu F=%zu)\n", geometry_post_dump,
               dv.size()/3, df.size()/3);
    }
    if (!do_bake) { printf("(--no-bake) done\n"); return 0; }

    VoxelPbr vox{&coords, &pbr6, res, do_snap ? &bvh : nullptr};
    const std::vector<float> no_vp;
    trellis::BakedMesh bm = boxuv
        ? trellis::uv_box_project(dv, (int)dv.size()/3, df, (int)df.size()/3, no_vp, atlas, &vox)
        : trellis::uv_bake(dv, (int)dv.size()/3, df, (int)df.size()/3, no_vp, atlas, &vox);
    if (!boxuv && !bm.ok())
        bm = trellis::uv_chart_project(dv, (int)dv.size()/3, df, (int)df.size()/3, no_vp, atlas, &vox);
    printf("  [bake %.1fs]\n", now()-t);
    if (!bm.ok()) { fprintf(stderr, "bake failed\n"); return 1; }
    printf("  [audit] bake: faces in=%zu out=%zu (dropped %lld)\n",
           df.size()/3, bm.faces.size()/3, (long long)(df.size()/3) - (long long)(bm.faces.size()/3));
    trellis::write_glb_textured(out, bm.verts.data(), (int64_t)bm.verts.size()/3, bm.uv.data(),
                                bm.faces.data(), (int64_t)bm.faces.size()/3, bm.base.data(), bm.mr.data(), bm.T,
                                /*double_sided=*/rm.F() == 0);
    printf("wrote %s (atlas %d)\n", out, bm.T);
    return 0;
}
