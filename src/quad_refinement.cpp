#include "quad_refinement.h"
#include "quad_quality.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <queue>
#include <stdexcept>

namespace trellis {

using V = std::array<double, 3>;

QuadRefinementResult refine_quads(
    const std::vector<std::array<float, 3>>& input_vertices,
    const std::vector<std::vector<uint32_t>>& input_faces,
    const QuadRefinementOptions& options) {
    if (!(options.center_normal_span_degrees > 0 && options.center_normal_span_degrees <= 180))
        throw std::invalid_argument("Invalid center normal span");
    if (input_vertices.size() > UINT32_MAX || input_faces.empty())
        throw std::invalid_argument("Invalid refinement input size");
    std::vector<V> vertices;
    vertices.reserve(input_vertices.size());
    for (const auto& p : input_vertices) {
        for (float x : p) if (!std::isfinite(x)) throw std::invalid_argument("Nonfinite vertex");
        vertices.push_back({p[0], p[1], p[2]});
    }
    std::vector<std::vector<size_t>> faces;
    faces.reserve(input_faces.size());
    for (const auto& f : input_faces) {
        if (f.size() != 3 && f.size() != 4) throw std::invalid_argument("Expected triangle or quad");
        std::vector<size_t> q;
        for (uint32_t index : f) {
            if (index >= vertices.size()) throw std::invalid_argument("Invalid face vertex");
            q.push_back(index);
        }
        faces.push_back(std::move(q));
    }
    const auto& forced = options.forced_faces;
    const auto& frozen = options.frozen_faces;
    const bool surface = options.surface;
    const bool constrain_normals = options.center_normal_span_degrees < 180;
    const double normalLimit = options.center_normal_span_degrees;
    if (faces.empty())
      throw std::runtime_error("Empty mesh");
    for (auto id : forced)
      if (id >= faces.size())
        throw std::runtime_error("Invalid forced polygon index");
    std::vector<std::vector<std::pair<size_t, double>>> witnesses;
    for (size_t i = 0; i < vertices.size(); ++i)
      witnesses.push_back({{i, 1}});
    using Edge = std::array<size_t, 2>;
    auto key = [](size_t a, size_t b) -> Edge {
      return {std::min(a, b), std::max(a, b)};
    };
    std::map<Edge, std::vector<size_t>> sides;
    std::set<Edge> split;
    for (size_t fi = 0; fi < faces.size(); ++fi) {
      auto &f = faces[fi];
      for (size_t j = 0; j < f.size(); ++j) {
        auto edge = key(f[j], f[(j + 1) % f.size()]);
        sides[edge].push_back(fi);
        if (f.size() == 3 || forced.count(fi))
          split.insert(edge);
      }
    }
    for (auto &[edge, fs] : sides)
      if (fs.size() != 2)
        throw std::runtime_error("Expected closed manifold input");
    if (options.use_split_seed) {
      split.clear();
      for (auto edge : options.split_seed) {
        edge = key(edge[0], edge[1]);
        if (!sides.count(edge)) throw std::runtime_error("Invalid seeded edge");
        split.insert(edge);
      }
      for (auto fi : forced) {
        auto &f = faces[fi];
        for (size_t j = 0; j < f.size(); ++j)
          split.insert(key(f[j], f[(j + 1) % f.size()]));
      }
    }
    std::set<Edge> frozenEdges;
    for (auto fi : frozen) {
      if (fi >= faces.size() || faces[fi].size() != 4 || forced.count(fi))
        throw std::runtime_error("Invalid frozen quad");
      auto &f = faces[fi];
      for (size_t j = 0; j < 4; ++j) {
        auto edge = key(f[j], f[(j + 1) % 4]);
        frozenEdges.insert(edge);
        split.erase(edge);
      }
    }
    std::set<size_t> odd;

    for (size_t fi = 0; fi < faces.size(); ++fi)
      if (!forced.count(fi) && !frozen.count(fi)) {
        size_t count = 0;
        auto &f = faces[fi];
        for (size_t j = 0; j < f.size(); ++j)
          count += split.count(key(f[j], f[(j + 1) % f.size()]));
        if ((count + f.size()) % 2) {
          odd.insert(fi);
        }
      }
    size_t paths = 0;
    while (!odd.empty()) {
      size_t start = *odd.begin(), target = faces.size();
      std::vector<size_t> prev(faces.size(), faces.size());
      std::vector<Edge> via(faces.size());
      std::queue<size_t> queue;
      queue.push(start);
      prev[start] = start;
      while (!queue.empty() && target == faces.size()) {
        size_t at = queue.front();
        queue.pop();
        auto &f = faces[at];
        for (size_t j = 0; j < f.size(); ++j) {
          auto edge = key(f[j], f[(j + 1) % f.size()]);
          if (frozenEdges.count(edge))
            continue;
          auto &fs = sides.at(edge);
          size_t next = fs[0] == at ? fs[1] : fs[0];
          if (frozen.count(next) || forced.count(next) ||
              prev[next] != faces.size())
            continue;
          prev[next] = at;
          via[next] = edge;
          if (odd.count(next)) {
            target = next;
            break;
          }
          queue.push(next);
        }
      }
      if (target == faces.size())
        throw std::runtime_error(
            "Odd split boundary cannot be paired without crossing fixed polygons");
      for (size_t at = target; at != start; at = prev[at]) {
        auto edge = via[at];
        if (split.count(edge))
          split.erase(edge);
        else
          split.insert(edge);
      }
      odd.erase(start);
      odd.erase(target);
      ++paths;
    }
    std::map<Edge, size_t> midpoints;
    std::vector<std::array<size_t, 4>> quads;
    std::vector<size_t> parent;
    std::map<size_t, size_t> centers;
    size_t unchanged = 0, opposite = 0, adjacent = 0, full = 0;
    for (size_t fi = 0; fi < faces.size(); ++fi) {
      auto &f = faces[fi];
      std::vector<size_t> marked;
      for (size_t j = 0; j < f.size(); ++j)
        if (split.count(key(f[j], f[(j + 1) % f.size()])))
          marked.push_back(j);
      auto emit = [&](std::array<size_t, 4> q) {
        quads.push_back(q);
        parent.push_back(fi);
      };
      if (marked.empty() && f.size() == 4) {
        emit({f[0], f[1], f[2], f[3]});
        ++unchanged;
        continue;
      }
      if (f.size() == 4 && marked.size() != 2 && marked.size() != 4)
        throw std::runtime_error("Invalid quad parity");
      if (f.size() == 3 && marked.size() != 3 && marked.size() != 1)
        throw std::runtime_error("Triangle boundary lost split");
      std::vector<size_t> mids(f.size(), size_t(-1));
      for (auto j : marked) {
        auto edge = key(f[j], f[(j + 1) % f.size()]);
        auto found = midpoints.find(edge);
        if (found == midpoints.end()) {
          V p;
          for (int k = 0; k < 3; ++k)
            p[k] = float((vertices[edge[0]][k] + vertices[edge[1]][k]) * .5);
          size_t id = vertices.size();
          vertices.push_back(p);
          witnesses.push_back({{edge[0], .5}, {edge[1], .5}});
          found = midpoints.emplace(edge, id).first;
        }
        mids[j] = found->second;
      }
      if (f.size() == 3 && marked.size() == 1) {
        size_t j = marked[0];
        std::array<size_t, 4> boundary = {f[j], mids[j], f[(j + 1) % 3],
                                          f[(j + 2) % 3]},
                              inner;
        for (size_t k = 0; k < 4; ++k) {
          std::map<size_t, double> weights;
          for (auto i : f)
            weights[i] += 1.0 / 6;
          for (auto entry : witnesses[boundary[k]])
            weights[entry.first] += .5 * entry.second;
          if (k == 1) {
            for (auto entry : witnesses[boundary[k]])
              weights[entry.first] += .1 * entry.second;
            weights[f[(j + 2) % 3]] -= .1;
          }
          V p = {0, 0, 0};
          for (auto entry : weights)
            for (int axis = 0; axis < 3; ++axis)
              p[axis] += vertices[entry.first][axis] * entry.second;
          for (auto &x : p)
            x = float(x);
          inner[k] = vertices.size();
          vertices.push_back(p);
          witnesses.emplace_back(weights.begin(), weights.end());
        }
        for (size_t k = 0; k < 4; ++k)
          emit({boundary[k], boundary[(k + 1) % 4], inner[(k + 1) % 4],
                inner[k]});
        emit(inner);
        continue;
      }
      if (f.size() == 4 && marked.size() == 2 && (marked[1] - marked[0]) == 2) {
        size_t j = marked[0];
        emit({f[j], mids[j], mids[(j + 2) % 4], f[(j + 3) % 4]});
        emit({mids[j], f[(j + 1) % 4], f[(j + 2) % 4], mids[(j + 2) % 4]});
        ++opposite;
        continue;
      }
      V center = {0, 0, 0};
      for (auto i : f)
        for (int k = 0; k < 3; ++k)
          center[k] += vertices[i][k] / f.size();
      for (auto &x : center)
        x = float(x);
      size_t c = vertices.size();
      vertices.push_back(center);
      witnesses.emplace_back();
      for (auto i : f)
        witnesses.back().push_back({i, 1.0 / f.size()});
      centers[fi] = c;
      if (surface && f.size() == 4) {
        for (int k = 0; k < 3; ++k)
          vertices[c][k] = float((vertices[f[0]][k] + vertices[f[2]][k]) * .5);
        witnesses[c] = {{f[0], .5}, {f[2], .5}};
      }
      if (marked.size() == f.size()) {
        for (size_t j = 0; j < f.size(); ++j)
          emit({f[j], mids[j], c, mids[(j + f.size() - 1) % f.size()]});
        ++full;
      } else {
        size_t first = 0;
        for (size_t j = 0; j < 4; ++j)
          if (mids[j] != size_t(-1) && mids[(j + 1) % 4] != size_t(-1))
            first = j;
        std::vector<size_t> boundary;
        for (size_t step = 0; step < 4; ++step) {
          size_t j = (first + step) % 4;
          boundary.push_back(f[j]);
          if (mids[j] != size_t(-1))
            boundary.push_back(mids[j]);
        }
        std::rotate(boundary.begin(), boundary.begin() + 1, boundary.end());
        V transition = {0, 0, 0};
        for (size_t j = 0; j < 6; j += 2)
          for (int k = 0; k < 3; ++k)
            transition[k] += vertices[boundary[j]][k] / 3;
        for (auto &x : transition)
          x = float(x);
        if (!surface) {
          vertices[c] = transition;
          witnesses[c].clear();
          for (size_t j = 0; j < 6; j += 2)
            for (auto entry : witnesses[boundary[j]])
              witnesses[c].push_back({entry.first, entry.second / 3});
        }
        for (size_t j = 0; j < 6; j += 2)
          emit({boundary[j], boundary[(j + 1) % 6], boundary[(j + 2) % 6], c});
        ++adjacent;
      }
    }
    auto qualityAt = [&](const std::array<size_t, 4> &q, size_t centerId,
                         const V &candidate) {
      std::array<trellis::QuadPoint, 4> p;
      for (int corner = 0; corner < 4; ++corner)
        p[corner] = q[corner] == centerId ? candidate : vertices[q[corner]];
      return trellis::quad_min_scaled_jacobian(p);
    };
    std::vector<double> quality;
    quality.reserve(quads.size());
    for (const auto &q : quads)
      quality.push_back(qualityAt(q, vertices.size(), {}));
    std::map<size_t, std::vector<size_t>> groups;
    for (size_t i = 0; i < quads.size(); ++i)
      groups[parent[i]].push_back(i);
    auto spanAt = [&](const std::array<size_t, 4> &q, size_t centerId,
                      const V &candidate) {
      std::array<V, 4> normals;
      for (int corner = 0; corner < 4; ++corner) {
        auto point = [&](size_t id) -> const V & {
          return id == centerId ? candidate : vertices[id];
        };
        V a{}, b{};
        for (int k = 0; k < 3; ++k) {
          a[k] = point(q[(corner + 1) % 4])[k] - point(q[corner])[k];
          b[k] = point(q[(corner + 3) % 4])[k] - point(q[corner])[k];
        }
        auto &n = normals[corner];
        n = {a[1] * b[2] - a[2] * b[1],
             a[2] * b[0] - a[0] * b[2],
             a[0] * b[1] - a[1] * b[0]};
        double length = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        if (!(length > 0)) return 180.0;
        for (auto &x : n) x /= length;
      }
      double minimum = 1;
      for (int i = 0; i < 4; ++i)
        for (int j = 0; j < i; ++j) {
          double d = 0;
          for (int k = 0; k < 3; ++k) d += normals[i][k] * normals[j][k];
          minimum = std::min(minimum, d);
        }
      return std::acos(std::clamp(minimum, -1.0, 1.0)) *
             180 / 3.141592653589793;
    };
    auto edgeRatioAt = [&](const std::array<size_t, 4> &q, size_t centerId,
                           const V &candidate) {
      double shortest = 1e100, longest = 0;
      for (int corner = 0; corner < 4; ++corner) {
        const V &a = q[corner] == centerId ? candidate : vertices[q[corner]];
        const V &b = q[(corner + 1) % 4] == centerId
                         ? candidate : vertices[q[(corner + 1) % 4]];
        double squared = 0;
        for (int k = 0; k < 3; ++k)
          squared += (a[k] - b[k]) * (a[k] - b[k]);
        shortest = std::min(shortest, squared);
        longest = std::max(longest, squared);
      }
      return shortest > 0 ? std::sqrt(longest / shortest) : 1e100;
    };
    size_t optimized = 0;
    for (auto &[fi, centerId] : centers) {
      auto &group = groups.at(fi);
      double best = 1;
      for (auto i : group)
        best = std::min(best, quality[i]);
      double currentSpan = 0;
      double currentRatio = 0;
      if (constrain_normals)
        for (auto i : group) {
          currentSpan = std::max(currentSpan,
                                 spanAt(quads[i], centerId, vertices[centerId]));
          currentRatio = std::max(currentRatio,
                                  edgeRatioAt(quads[i], centerId,
                                              vertices[centerId]));
        }
      const bool repairSpan = constrain_normals && currentSpan > normalLimit;
      if ((best >= .1 && !repairSpan) || faces[fi].size() != 4)
        continue;
      double bestScore = best;
      if (repairSpan)
        bestScore = best >= .1 ? -currentSpan : -1e30;
      auto &f = faces[fi];
      std::vector<V> positions;
      std::vector<std::vector<std::pair<size_t, double>>> positionWitnesses;
      if (surface) {
        for (int j = 1; j < 20; ++j) {
          double t = j * .05;
          V p;
          for (int k = 0; k < 3; ++k)
            p[k] = float((1 - t) * vertices[f[0]][k] + t * vertices[f[2]][k]);
          positions.push_back(p);
          positionWitnesses.push_back({{f[0], 1 - t}, {f[2], t}});
        }
      } else
        for (int u = 1; u <= 9; ++u)
          for (int v = 1; v <= 9; ++v) {
            double x = u * .1, y = v * .1;
            V p;
            for (int k = 0; k < 3; ++k)
              p[k] = float((1 - x) * (1 - y) * vertices[f[0]][k] +
                           x * (1 - y) * vertices[f[1]][k] +
                           x * y * vertices[f[2]][k] +
                           (1 - x) * y * vertices[f[3]][k]);
            positions.push_back(p);
            positionWitnesses.push_back({{f[0], (1 - x) * (1 - y)},
                                         {f[1], x * (1 - y)},
                                         {f[2], x * y},
                                         {f[3], (1 - x) * y}});
          }
      std::vector<double> qs;
      qs.reserve(positions.size() * group.size());
      for (const auto &position : positions)
        for (auto qi : group)
          qs.push_back(qualityAt(quads[qi], centerId, position));
      size_t chosen = positions.size();
      for (size_t pi = 0; pi < positions.size(); ++pi) {
        double score = 1;
        for (size_t j = 0; j < group.size(); ++j)
          score = std::min(score, qs[pi * group.size() + j]);
        double objective = score;
        if (repairSpan) {
          if (score < .1) continue;
          double candidateSpan = 0;
          double candidateRatio = 0;
          for (auto qi : group) {
            candidateSpan = std::max(candidateSpan,
                                     spanAt(quads[qi], centerId, positions[pi]));
            candidateRatio = std::max(candidateRatio,
                                      edgeRatioAt(quads[qi], centerId,
                                                  positions[pi]));
          }
          if (candidateRatio > std::max(100.0, currentRatio) + 1e-8)
            continue;
          objective = -candidateSpan;
        }
        if (objective > bestScore + 1e-12) {
          bestScore = objective;
          best = score;
          chosen = pi;
        }
      }
      if (chosen != positions.size()) {
        vertices[centerId] = positions[chosen];
        witnesses[centerId] = positionWitnesses[chosen];
        for (size_t j = 0; j < group.size(); ++j)
          quality[group[j]] = qs[chosen * group.size() + j];
        ++optimized;
      }
    }
    size_t rotated = 0, uncovered = 0;
    if (surface)
      for (size_t qi = 0; qi < quads.size(); ++qi) {
        auto &q = quads[qi];
        auto &f = faces[parent[qi]];
        auto score = [&](size_t offset) {
          int count = 0;
          for (size_t j = 1; j < 3; ++j) {
            std::set<size_t> support;
            for (auto vi :
                 {q[offset % 4], q[(offset + j) % 4], q[(offset + j + 1) % 4]})
              for (auto entry : witnesses[vi])
                if (entry.second > 0)
                  support.insert(entry.first);
            for (size_t k = 1; k + 1 < f.size(); ++k) {
              std::set<size_t> reference = {f[0], f[k], f[k + 1]};
              if (std::includes(reference.begin(), reference.end(),
                                support.begin(), support.end())) {
                ++count;
                break;
              }
            }
          }
          return count;
        };
        int a = score(0), b = score(1);
        if (b > a) {
          std::rotate(q.begin(), q.begin() + 1, q.end());
          ++rotated;
        }
        if (std::max(a, b) < 2)
          ++uncovered;
      }
    QuadRefinementResult result;
    if (vertices.size() > UINT32_MAX) throw std::overflow_error("Too many refined vertices");
    result.vertices.reserve(vertices.size());
    for (const auto& p : vertices)
        result.vertices.push_back({float(p[0]), float(p[1]), float(p[2])});
    result.quads.reserve(quads.size());
    for (const auto& q : quads)
        result.quads.push_back({uint32_t(q[0]), uint32_t(q[1]), uint32_t(q[2]), uint32_t(q[3])});
    result.parents = std::move(parent);
    result.quality = std::move(quality);
    result.vertex_witnesses = std::move(witnesses);
    result.split_edges = std::move(split);
    result.parity_paths = paths;
    result.unchanged = unchanged;
    result.opposite = opposite;
    result.adjacent = adjacent;
    result.full = full;
    result.optimized_centers = optimized;
    result.rotated_for_source_coverage = rotated;
    result.unproven_quads = uncovered;
    return result;
}

double quad_corner_normal_span_degrees(
    const std::array<std::array<float, 3>, 4>& points) {
    std::array<V, 4> normals;
    for (int corner = 0; corner < 4; ++corner) {
        V a{}, b{};
        for (int axis = 0; axis < 3; ++axis) {
            a[axis] = double(points[(corner+1)%4][axis]) - points[corner][axis];
            b[axis] = double(points[(corner+3)%4][axis]) - points[corner][axis];
        }
        auto& n = normals[corner];
        n = {a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]};
        const double length = std::sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]);
        if (!(length > 0)) return 180.0;
        for (double& value : n) value /= length;
    }
    double minimum = 1;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < i; ++j) {
            double dot = 0;
            for (int axis = 0; axis < 3; ++axis) dot += normals[i][axis]*normals[j][axis];
            minimum = std::min(minimum, dot);
        }
    return std::acos(std::clamp(minimum, -1.0, 1.0))*180/3.141592653589793;
}

}
