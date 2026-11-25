/*
 *  geometry2d_oracle.h
 *
 *  This file is part of NEST GPU.
 *
 *  Copyright (C) 2021 The NEST Initiative
 *
 *  NEST GPU is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  NEST GPU is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with NEST GPU.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef GEOMETRY2D_ORACLE_H
#define GEOMETRY2D_ORACLE_H

// 2D exact oracle for geometry_oracle.h. The shapes are rebuilt from the mask
// parameters with the documented conventions only (no SAPI geometry code):
//   circle {r}; ellipse {a, b[, degrees]}, semi-axis a along the rotated x
//   axis; parallelogram {a0, a1}, vertices -a0, -a1, a0, a1; triangle {a0, a1},
//   vertices 0, a0, a1, all relative to the anchor; annulus {r_in, r_out}, the
//   closed ring r_in <= |d| <= r_out (not convex, so never tested as a convex
//   set: a convex polygon meets it iff its farthest vertex is at least r_in
//   from the anchor and its nearest point at most r_out, since the distance
//   to the anchor takes every value in between on the connected polygon).
// Two independent deciders:
//   analytic: an ellipse maps to the unit disk by its inverse affine map, which
//             keeps the region convex, so it meets the region iff the mapped
//             region holds the origin or lies within distance 1 of it;
//             polygons are separated by an edge normal of either, which
//             gives a signed separation to compare with the margin.
//   sampled:  dense boundary points of each set tested against the other.

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <vector>

#include "geometry_oracle.h"
#include "geometry_test_helpers.h"

namespace geometry_test {
struct Shape2D {
  sapi::MASK_SHAPE kind = sapi::MASK_SHAPE::NULL_MS;
  // Ellipse (circles too): semi-axes and the cosine and sine of the rotation
  double a = 0, b = 0, cos = 1, sin = 0;
  // Parallelogram and triangle: counter-clockwise vertices
  std::vector<P> vertices;
  // Annulus: a = b = r_out
  double r_in = 0;

  bool is_annulus() const { return kind == sapi::MASK_SHAPE::ANNULAR; }

  bool is_round() const {
    return kind == sapi::MASK_SHAPE::CIRCULAR ||
           kind == sapi::MASK_SHAPE::ELLIPTICAL;
  }

  // Coordinates in the frame of the ellipse axes, scaled to the unit circle
  P to_unit(const P d, const double scale) const {
    return {(cos * d.x + sin * d.y) / (a * scale),
            (-sin * d.x + cos * d.y) / (b * scale)};
  }
};

// Largest distance from the anchor to the shape
inline double shape_extent(const Shape2D &s) {
  if (s.is_round() || s.is_annulus())
    return std::max(s.a, s.b);
  double size = 0;
  for (const auto &p : s.vertices)
    size = std::max(size, norm(p));
  return size;
}

// Polygon vertices scaled about the anchor
inline std::vector<P> scaled_vertices(const Shape2D &s, const double scale) {
  std::vector<P> v;
  for (const auto &p : s.vertices)
    v.emplace_back(p * scale);
  return v;
}

inline bool contains(const Shape2D &s, const double scale, const P d) {
  if (s.is_annulus()) {
    const double n = norm(d);
    return n <= s.b * scale && n >= s.r_in * scale;
  }
  if (s.is_round()) {
    const P u = s.to_unit(d, scale);
    return dot(u, u) <= 1;
  }
  return in_convex(d, scaled_vertices(s, scale));
}

// Point verdict outside the tolerance band: 1 robustly inside, -1 robustly
// outside, 0 within ORACLE_MARGIN (relative to the shape size) of the
// boundary. Scaling about the anchor would not move the polygon edges through
// the anchor (triangle), so polygons use the distance to the perimeter.
inline int classify(const Shape2D &s, const P d) {
  if (s.is_annulus()) {
    const double n = norm(d), band = ORACLE_MARGIN * shape_extent(s);
    if (std::abs(n - s.b) <= band ||
        (s.r_in > 0 && std::abs(n - s.r_in) <= band))
      return 0;
    return contains(s, 1, d) ? 1 : -1;
  }
  if (s.is_round())
    return contains(s, 1 - ORACLE_MARGIN, d)    ? 1
           : !contains(s, 1 + ORACLE_MARGIN, d) ? -1
                                                : 0;

  if (perimeter_distance(d, s.vertices) <= ORACLE_MARGIN * shape_extent(s))
    return 0;
  return contains(s, 1, d) ? 1 : -1;
}

// Largest gap between the projections of u and v on the unit edge normals of
// u (negative when they overlap on every normal)
inline double edge_separation(const std::vector<P> &u,
                              const std::vector<P> &v) {
  double best = -std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < u.size(); ++i) {
    const P e = u[(i + 1) % u.size()] - u[i];
    const P n = P{e.y, -e.x} * (1 / norm(e));
    double u_min = std::numeric_limits<double>::infinity(), u_max = -u_min;
    double v_min = u_min, v_max = -u_min;
    for (const auto &p : u) {
      const double pn = dot(p, n);
      u_min = std::min(u_min, pn);
      u_max = std::max(u_max, pn);
    }
    for (const auto &p : v) {
      const double pn = dot(p, n);
      v_min = std::min(v_min, pn);
      v_max = std::max(v_max, pn);
    }
    best = std::max(best, std::max(v_min - u_max, u_min - v_max));
  }
  return best;
}

// Signed separation of two convex polygons: their distance when disjoint
// (an edge normal of either separates them), otherwise minus their
// penetration depth (the smallest overlap over all edge normals)
inline double polygon_separation(const std::vector<P> &u,
                                 const std::vector<P> &v) {
  const double gap = std::max(edge_separation(u, v), edge_separation(v, u));
  if (gap <= 0)
    return gap;

  // The closest points of disjoint convex polygons include a vertex of one
  double distance = std::numeric_limits<double>::infinity();
  for (const auto &p : u)
    distance = std::min(distance, perimeter_distance(p, v));
  for (const auto &p : v)
    distance = std::min(distance, perimeter_distance(p, u));
  return distance;
}

// Whether the ellipse scaled about its center meets the convex region: its
// inverse affine map takes the ellipse to the unit disk and keeps the region
// convex
inline bool ellipse_meets(const Shape2D &s, const double scale,
                          const std::vector<P> &region) {
  std::vector<P> unit;
  for (const auto &p : region)
    unit.emplace_back(s.to_unit(p, scale));
  const P origin{0, 0};
  return in_convex(origin, unit) || perimeter_distance(origin, unit) <= 1;
}

// Analytic decider. Ellipses scale about their center (the anchor); polygons
// use the separation against an absolute margin, since scaling about the
// anchor would not move the triangle edges through its anchor vertex.
inline OracleVerdict shape_verdict(const Shape2D &s,
                                   const std::vector<P> &region) {
  if (s.is_annulus()) {
    // The ring shrunk or grown by an absolute margin on both circles
    const double delta = ORACLE_MARGIN * shape_extent(s);
    double farthest = 0;
    for (const auto &p : region)
      farthest = std::max(farthest, norm(p));
    const double nearest = region_distance({0, 0}, region);
    const double inner_shrunk = s.r_in > 0 ? s.r_in + delta : 0;
    const double inner_grown = s.r_in > 0 ? s.r_in - delta : 0;
    return {inner_shrunk <= s.b - delta && farthest >= inner_shrunk &&
                nearest <= s.b - delta,
            farthest >= inner_grown && nearest <= s.b + delta};
  }
  if (s.is_round())
    return {ellipse_meets(s, 1 - ORACLE_MARGIN, region),
            ellipse_meets(s, 1 + ORACLE_MARGIN, region)};

  const double margin = ORACLE_MARGIN * shape_extent(s);
  const double separation = polygon_separation(s.vertices, region);
  return {separation < -margin, separation <= margin};
}

// Points of the shape boundary (relative to the anchor), uniform in the
// ellipse angle or in polygon perimeter length
inline std::vector<P> boundary_points(const Shape2D &s, const int count) {
  std::vector<P> points;
  if (s.is_annulus()) {
    for (const double r : {s.b, s.r_in}) {
      if (r <= 0)
        continue;
      for (int k = 0; k < count; ++k) {
        const double t = 2 * std::numbers::pi * k / count;
        points.emplace_back(P{r * std::cos(t), r * std::sin(t)});
      }
    }
    return points;
  }
  if (s.is_round()) {
    for (int k = 0; k < count; ++k) {
      const double t = 2 * std::numbers::pi * k / count;
      const double x = s.a * std::cos(t), y = s.b * std::sin(t);
      points.emplace_back(s.cos * x - s.sin * y, s.sin * x + s.cos * y);
    }
    return points;
  }

  for (const auto &c : sample_region(s.vertices, 0, count))
    points.emplace_back(from(c));
  return points;
}

// Sampled decider: a boundary point of one set lies in the other. Each hit is
// a real common point, so it never reports an overlap the shape (grown by any
// margin) does not have; it may miss thin overlaps.
inline bool sampled_meets(const Shape2D &s, const std::vector<P> &region,
                          const int count) {
  for (const auto &p : boundary_points(s, count))
    if (in_convex(p, region))
      return true;
  for (const auto &c : sample_region(region, 0, count))
    if (contains(s, 1, from(c)))
      return true;
  return false;
}

template <> struct GeometryOracle<sapi::Coord2D> {
  using Shape = Shape2D;
  using Region = std::vector<P>;

  static Shape shape(const MaskDef &def) {
    Shape s;
    s.kind = def.shape;
    const auto &p = def.params;
    switch (def.shape) {
    case sapi::MASK_SHAPE::CIRCULAR:
      s.a = s.b = p[0];
      break;
    case sapi::MASK_SHAPE::ELLIPTICAL: {
      s.a = p[0];
      s.b = p[1];
      const double angle = p.size() > 2 ? p[2] * std::numbers::pi / 180 : 0.;
      // The ellipse turns clockwise by the angle
      s.cos = std::cos(angle);
      s.sin = -std::sin(angle);
      break;
    }
    case sapi::MASK_SHAPE::ANNULAR:
      s.r_in = p[0];
      s.a = s.b = p[1];
      break;
    case sapi::MASK_SHAPE::PARALLELOGRAM:
      s.vertices = convex_hull(
          {{-p[0], -p[1]}, {-p[2], -p[3]}, {p[0], p[1]}, {p[2], p[3]}});
      break;
    case sapi::MASK_SHAPE::TRIANGULAR:
      s.vertices = convex_hull({{0, 0}, {p[0], p[1]}, {p[2], p[3]}});
      break;
    default:
      require(false, "unknown mask shape");
    }
    return s;
  }

  // Definition of -shape: the centered shapes are symmetric, the triangle flips
  static MaskDef reflected(const MaskDef &def) {
    MaskDef r = def;
    if (def.shape == sapi::MASK_SHAPE::TRIANGULAR)
      for (auto &v : r.params)
        v = -v;
    return r;
  }

  static Region tile_region(const sapi::Tile<sapi::Coord2D> &tile,
                            const sapi::Coord2D &anchor) {
    Region r;
    for (const auto &v : polygon(tile))
      r.emplace_back(v - from(anchor));
    return r;
  }

  static Region pair_region(const sapi::Tile<sapi::Coord2D> &a,
                            const sapi::Tile<sapi::Coord2D> &b,
                            const sapi::Coord2D &anchor) {
    return minkowski_difference(a, b, from(anchor));
  }

  static Region point_region(const sapi::Coord2D &coord,
                             const sapi::Tile<sapi::Coord2D> &b,
                             const sapi::Coord2D &anchor) {
    Region r;
    for (const auto &v : polygon(b))
      r.emplace_back(v - from(coord) - from(anchor));
    return r;
  }

  static OracleVerdict verdict(const Shape &s, const Region &r) {
    return shape_verdict(s, r);
  }

  static double extent(const Shape &s) { return shape_extent(s); }

  static std::vector<sapi::Coord2D>
  node_samples(const sapi::Tile<sapi::Coord2D> &tile) {
    return sample_region(polygon(tile), 6, 24);
  }
};
} // namespace geometry_test

#endif
