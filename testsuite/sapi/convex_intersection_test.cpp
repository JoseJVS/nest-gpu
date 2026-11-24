/*
 *  convex_intersection_test.cpp
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

// The exact intersection primitives behind the mask/tile filters:
//   Tile::support, farthest_point    vertex supports, in any vertex order
//   mask_support                     2D mask shapes against their boundary
//   closest_point_on_simplex_to_origin  against the optimality condition of
//                                    the closest point (2D and 3D)
//   convex_sets_intersect            GJK against the independent oracle of
//                                    geometry2d_oracle.h (separating axes for
//                                    polygons, inverse affine map for
//                                    ellipses), touching and gaps at the
//                                    margin, the uncertain paths, and 3D
//                                    spheres and boxes

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <deque>
#include <limits>
#include <numbers>
#include <random>
#include <string>
#include <vector>

#include "convex_intersection.h"
#include "coordinate_geometry.h"
#include "geometry2d_oracle.h"
#include "geometry_oracle.h"
#include "geometry_test_helpers.h"
#include "mask.h"
#include "tile.h"

using namespace sapi;
using namespace geometry_test;

namespace {
using Oracle = GeometryOracle<Coord2D>;
using Rng = std::mt19937_64;

double uniform(Rng &rng, const double lo, const double hi) {
  return std::uniform_real_distribution<double>(lo, hi)(rng);
}

P random_point(Rng &rng, const double reach) {
  return {uniform(rng, -reach, reach), uniform(rng, -reach, reach)};
}

bool finite(const Coord2D &c) {
  return std::isfinite(c.x_) && std::isfinite(c.y_);
}

bool finite(const Coord3D &c) {
  return std::isfinite(c.x_) && std::isfinite(c.y_) && std::isfinite(c.z_);
}

// Circles, ellipses rotated every 15 degrees with aspect ratios up to 1:100,
// sheared, thin and negative parallelograms, triangles of both orientations
std::vector<MaskDef> mask_defs() {
  std::vector<MaskDef> defs{{MASK_SHAPE::CIRCULAR, {0.35}},
                            {MASK_SHAPE::CIRCULAR, {2.}},
                            {MASK_SHAPE::ELLIPTICAL, {0.4, 1.2}}};
  for (int degrees = 0; degrees < 360; degrees += 15) {
    defs.push_back({MASK_SHAPE::ELLIPTICAL, {2., 0.5, double(degrees)}});
    defs.push_back({MASK_SHAPE::ELLIPTICAL, {1.5, 0.015, double(degrees)}});
  }
  for (const auto &params :
       std::vector<std::vector<space_t>>{{1., 0.2, -0.3, 0.8},
                                         {0.8, 0., 0.2, 0.4},
                                         {1., 0., 1., 1e-2},
                                         {-0.5, -0.5, 0.5, -0.5}})
    defs.push_back({MASK_SHAPE::PARALLELOGRAM, params});
  for (const auto &params :
       std::vector<std::vector<space_t>>{{1.2, 0., 0., 1.2},
                                         {0., 0.9, 0.9, 0.},
                                         {1.5, 0.1, -0.4, 0.3},
                                         {2., 0.01, 1.9, 0.05},
                                         {-1., -0.5, -0.2, -1.1}})
    defs.push_back({MASK_SHAPE::TRIANGULAR, params});
  return defs;
}

std::string describe(const MaskDef &def) {
  std::string s = std::to_string(static_cast<int>(def.shape)) + " {";
  for (const auto p : def.params)
    s += std::to_string(p) + " ";
  return s + "}";
}

// Unit directions every 7.5 degrees (axes and edge normals included, so ties
// occur)
std::vector<P> directions() {
  std::vector<P> d;
  for (int k = 0; k < 48; ++k) {
    const double t = 2 * std::numbers::pi * k / 48;
    d.emplace_back(std::cos(t), std::sin(t));
  }
  return d;
}

std::vector<Coord2D> to_coords(const std::vector<P> &points) {
  std::vector<Coord2D> c;
  for (const auto &p : points)
    c.emplace_back(to_coord(p));
  return c;
}

// Tile::support returns a vertex reaching as far as every vertex and every
// sampled point, in either vertex order
void test_tile_support(const std::vector<NamedTile> &catalog) {
  for (const auto &nt : catalog) {
    const auto &tile = *nt.tile;
    std::vector<Coord2D> reversed(tile.vertices_.rbegin(),
                                  tile.vertices_.rend());
    const Tile<Coord2D> mirror(TILE_SHAPE(tile.shape_),
                               CircumscribedRadius<Coord2D>(tile.c_radius_),
                               std::move(reversed));
    const auto samples = sample_region(polygon(tile), 6, 24);
    const double scale =
        std::sqrt(tile.c_radius_.radius2_) + norm(from(tile.c_radius_.origin_));

    for (const auto &u : directions()) {
      const auto s = tile.support(to_coord(u));
      bool vertex = false;
      double reach = -std::numeric_limits<double>::infinity();
      for (const auto &v : tile.vertices_) {
        vertex = vertex || v == s;
        reach = std::max(reach, dot(from(v), u));
      }
      require(vertex, "support is not a vertex: " + nt.name);
      require(std::fabs(dot(from(s), u) - reach) <= 1e-15 * scale,
              "support short of a vertex: " + nt.name);
      require(std::fabs(dot(from(mirror.support(to_coord(u))), u) - reach) <=
                  1e-15 * scale,
              "support depends on the vertex order: " + nt.name);
      for (const auto &c : samples)
        require(dot(from(c), u) <= reach + 1e-12 * scale,
                "a tile point lies beyond the support: " + nt.name);
    }
  }
}

// mask_support lies on the shape and reaches as far as its densely sampled
// boundary, for absolute masks and blueprints, independently of the
// direction's length
void test_mask_support() {
  for (const auto &def : mask_defs()) {
    const auto shape = Oracle::shape(def);
    const double extent = shape_extent(shape);
    const auto boundary = boundary_points(shape, 20000);
    // Boundary sampling misses at most this much of an ellipse's reach
    const double sampling = shape.is_round() ? 1e-6 * extent : 1e-12 * extent;

    for (const bool absolute : {true, false}) {
      const auto mask = construct_mask<Coord2D>(
          def.shape,
          absolute ? std::vector<space_t>{0.3, -0.7} : std::vector<space_t>{},
          def.params, {-0.2, 0.4});
      const P anchor = from(mask.anchor_);

      for (const auto &u : directions()) {
        const P s = from(mask_support(mask, to_coord(u))) - anchor;
        double sampled = -std::numeric_limits<double>::infinity();
        for (const auto &b : boundary)
          sampled = std::max(sampled, dot(b, u));

        const double reach = dot(s, u);
        require(sampled - 1e-12 * extent <= reach,
                "support short of the boundary: " + describe(def));
        require(reach <= sampled + sampling,
                "support beyond the boundary: " + describe(def));
        require(classify(shape, s) >= 0,
                "support outside the shape: " + describe(def));

        for (const double length : {1e-12, 1e9}) {
          const P scaled = from(mask_support(mask, to_coord(u * length)));
          require(std::fabs(dot(scaled - anchor, u) - reach) <= 1e-12 * extent,
                  "support depends on the direction length: " + describe(def));
        }
      }
    }
  }
}

// Closest point q of the simplex hull to the origin, checked by the
// optimality condition (every vertex p has p.q >= q.q) and by q lying in the
// hull of the vertices kept. Returns the number of vertices kept.
template <Coordinate CoordT>
dim_t check_simplex(const std::vector<CoordT> &points, const bool degenerate,
                    const std::string &where) {
  Simplex<CoordT> simplex;
  for (const auto &p : points)
    simplex.push(p);
  const CoordT q = closest_point_on_simplex_to_origin(simplex);
  require(finite(q), "non-finite closest point: " + where);
  if (degenerate && simplex.size_ == CoordT::D + 1)
    return simplex.size_;

  double scale = 0;
  for (const auto &p : points)
    scale = std::max(scale, vector_norm(p));
  const double tol = 1e-12 * scale;

  for (const auto &p : points)
    require(vector_dot(p, q) >= vector_dot(q, q) - tol * scale,
            "not the closest point: " + where);

  const auto &kept = simplex.points_;
  switch (simplex.size_) {
  case 1:
    require(vector_norm(q - kept[0]) <= tol, "point feature: " + where);
    break;
  case 2: {
    const CoordT ab = kept[1] - kept[0];
    const double t = vector_dot(q - kept[0], ab) / vector_norm2(ab);
    require(-1e-12 <= t && t <= 1 + 1e-12 &&
                vector_norm(kept[0] + ab * t - q) <= tol,
            "segment feature: " + where);
    break;
  }
  default:
    if constexpr (CoordT::D == 3) {
      if (simplex.size_ == 3) {
        const Coord3D n = vector_cross(kept[1] - kept[0], kept[2] - kept[0]);
        const double area = vector_norm(n);
        require(std::fabs(vector_dot(q - kept[0], n)) <= tol * area,
                "triangle feature off its plane: " + where);
        for (int e = 0; e < 3; ++e)
          require(
              vector_dot(vector_cross(kept[(e + 1) % 3] - kept[e], q - kept[e]),
                         n) >= -tol * scale * area,
              "triangle feature outside its triangle: " + where);
        break;
      }
    }
    // Full simplex: it holds the origin, which is the closest point
    require(vector_norm(q) <= tol, "full simplex: " + where);
  }
  return simplex.size_;
}

double orient3(const Coord3D &a, const Coord3D &b, const Coord3D &c,
               const Coord3D &d) {
  return vector_dot(b - a, vector_cross(c - a, d - a));
}

void test_simplex() {
  Rng rng(2026);
  std::array<long, 5> sizes_2d{}, sizes_3d{};

  for (int k = 0; k < 20000; ++k) {
    // Near the origin often enough to hold it
    const double reach = k % 4 ? 1. : 3.;
    const P shift = random_point(rng, k % 2 ? 0.5 : 2.);
    std::vector<Coord2D> points;
    for (std::size_t i = 0; i < 2 + std::size_t(k % 2); ++i)
      points.emplace_back(to_coord(random_point(rng, reach) + shift));
    const auto kept = check_simplex(points, false, "2D random");
    ++sizes_2d[kept];
    if (kept == 3)
      require(in_convex({0, 0}, convex_hull({from(points[0]), from(points[1]),
                                             from(points[2])})),
              "2D triangle kept without the origin");

    std::vector<Coord3D> points3;
    for (std::size_t i = 0; i < 2 + std::size_t(k % 3); ++i)
      points3.emplace_back(construct_coord_3D(uniform(rng, -reach, reach),
                                              uniform(rng, -reach, reach),
                                              uniform(rng, -reach, reach)) +
                           construct_coord_3D(shift.x, shift.y, shift.x));
    const auto kept3 = check_simplex(points3, false, "3D random");
    ++sizes_3d[kept3];
    if (kept3 == 4) {
      const Coord3D o;
      const double volume =
          orient3(points3[0], points3[1], points3[2], points3[3]);
      for (std::size_t i = 0; i < 4; ++i) {
        auto replaced = points3;
        replaced[i] = o;
        require(orient3(replaced[0], replaced[1], replaced[2], replaced[3]) *
                        volume >=
                    0,
                "3D tetrahedron kept without the origin");
      }
    }
  }
  for (dim_t s = 1; s <= 3; ++s)
    require(sizes_2d[s] > 0, "a 2D simplex reduction never occurs");
  for (dim_t s = 1; s <= 4; ++s)
    require(sizes_3d[s] > 0, "a 3D simplex reduction never occurs");

  // Degenerate simplices: duplicate and collinear vertices, flat tetrahedra
  const auto c2 = [](double x, double y) { return construct_coord_2D(x, y); };
  const auto c3 = [](double x, double y, double z) {
    return construct_coord_3D(x, y, z);
  };
  check_simplex<Coord2D>({c2(1, 1), c2(1, 1)}, true, "duplicate segment");
  check_simplex<Coord2D>({c2(1, 1), c2(1, 1), c2(-1, 2)}, true,
                         "duplicate triangle vertex");
  check_simplex<Coord2D>({c2(1, 1), c2(1, 1), c2(1, 1)}, true,
                         "triangle of one point");
  check_simplex<Coord2D>({c2(-1, 1), c2(0, 1), c2(2, 1)}, true,
                         "collinear triangle off the origin");
  check_simplex<Coord2D>({c2(-1, 0), c2(1, 0), c2(2, 0)}, true,
                         "collinear triangle through the origin");
  check_simplex<Coord3D>({c3(1, 0, 1), c3(0, 1, 1), c3(1, 1, 1), c3(2, 2, 1)},
                         true, "flat tetrahedron");
  check_simplex<Coord3D>({c3(1, 1, 1), c3(1, 1, 1), c3(1, 1, 1), c3(0, 1, 1)},
                         true, "tetrahedron with duplicates");
  check_simplex<Coord3D>({c3(-1, 0, 0), c3(1, 0, 0), c3(0, 1, 0), c3(0, 2, 0)},
                         true, "flat tetrahedron through the origin");
}

// Support function of a point set (its convex hull)
struct HullSupport {
  std::vector<Coord2D> points;
  Coord2D operator()(const Coord2D &u) const {
    return farthest_point(points, u);
  }
};

// GJK on random convex polygons against their signed separation (separating
// axes): true below the margin, false above it, outside a rounding band
void test_polygons() {
  Rng rng(2026);
  long intersecting = 0, separated = 0;
  for (int k = 0; k < 20000; ++k) {
    std::vector<P> a, b;
    const P ca = random_point(rng, 1.), cb = random_point(rng, 3.);
    for (int i = 0; i < 3 + k % 6; ++i) {
      a.emplace_back(ca + random_point(rng, 1.));
      b.emplace_back(cb + random_point(rng, 0.2 + k % 3));
    }
    a = convex_hull(a);
    b = convex_hull(b);
    if (a.size() < 3 || b.size() < 3)
      continue;

    const double margin = k % 3 ? 0. : 0.05;
    const double separation = polygon_separation(a, b);
    const bool got = convex_sets_intersect(HullSupport{to_coords(a)},
                                           HullSupport{to_coords(b)},
                                           to_coord(ca - cb), margin);
    const double band = 1e-9;
    if (separation <= margin - band) {
      require(got, "GJK misses an intersection");
      ++intersecting;
    } else if (margin + band < separation) {
      require(!got, "GJK reports a gap as an intersection");
      ++separated;
    }
  }
  require(intersecting > 1000 && separated > 1000, "one-sided polygon cases");
}

// GJK on mask and tile supports against the oracle verdicts: source/target
// masks against tiles (B2) and blueprints against tile pairs (B1, the pair
// support is that of B - A)
void test_masks_and_tiles(const std::vector<NamedTile> &catalog) {
  Rng rng(2026);
  long robust = 0, apart = 0;
  for (const auto &def : mask_defs()) {
    const auto shape = Oracle::shape(def);
    const double extent = shape_extent(shape);
    for (const auto &nt : catalog) {
      if (std::count(nt.name.begin(), nt.name.end(), '.') > 1)
        continue; // roots and their first sub-tiles only
      const auto &tile = *nt.tile;
      const P centre = from(tile.c_radius_.origin_);
      const double reach = std::sqrt(tile.c_radius_.radius2_) + extent;
      for (int k = 0; k < 6; ++k) {
        const P origin = centre + random_point(rng, reach);
        const auto mask = construct_mask<Coord2D>(
            def.shape, {origin.x, origin.y}, def.params, {});
        const auto v = oracle_verdict<Coord2D>(
            shape, Oracle::tile_region(tile, mask.anchor_));
        const bool got = convex_sets_intersect(
            [&](const Coord2D &u) { return mask_support(mask, u); },
            [&](const Coord2D &u) { return tile.support(u); },
            mask.bounding_disc_.origin_ - tile.c_radius_.origin_, 0.);
        require(!v.robust || got, "GJK misses a mask/tile overlap: " +
                                      describe(def) + " " + nt.name);
        require(v.possible || !got, "GJK accepts a mask/tile gap: " +
                                        describe(def) + " " + nt.name);
        robust += v.robust;
        apart += !v.possible;
      }
    }
  }

  const auto roots = [&] {
    std::vector<const Tile<Coord2D> *> r;
    for (const auto &nt : catalog)
      if (nt.name.find('.') == std::string::npos)
        r.push_back(nt.tile);
    return r;
  }();
  for (const auto &def : mask_defs()) {
    const auto shape = Oracle::shape(def);
    for (const auto *a : roots)
      for (const auto *b : roots)
        for (int k = 0; k < 4; ++k) {
          const P ab = from(b->c_radius_.origin_) - from(a->c_radius_.origin_);
          const P offset =
              ab + random_point(rng, 2. + shape_extent(Oracle::shape(def)));
          const auto bp = construct_mask<Coord2D>(def.shape, {}, def.params,
                                                  {offset.x, offset.y});
          const auto v = oracle_verdict<Coord2D>(
              shape, Oracle::pair_region(*a, *b, bp.anchor_));
          const bool got = convex_sets_intersect(
              [&](const Coord2D &u) { return mask_support(bp, u); },
              [&](const Coord2D &u) {
                return b->support(u) - a->support(u.negative());
              },
              bp.bounding_disc_.origin_ -
                  (b->c_radius_.origin_ - a->c_radius_.origin_),
              0.);
          require(!v.robust || got,
                  "GJK misses a blueprint pair overlap: " + describe(def));
          require(v.possible || !got,
                  "GJK accepts a blueprint pair gap: " + describe(def));
          robust += v.robust;
          apart += !v.possible;
        }
  }
  require(robust > 1000 && apart > 1000, "one-sided mask/tile cases");
}

Tile<Coord2D> rectangle(const double x, const double y,
                        const std::vector<space_t> &sides) {
  return Tile<Coord2D>(TILE_SHAPE::RECTANGLE, construct_coord_2D(x, y), sides,
                       {});
}

bool tile_pair_meets(const Tile<Coord2D> &a, const Tile<Coord2D> &b,
                     const double margin) {
  return convex_sets_intersect([&](const Coord2D &u) { return a.support(u); },
                               [&](const Coord2D &u) { return b.support(u); },
                               a.c_radius_.origin_ - b.c_radius_.origin_,
                               margin);
}

bool mask_meets(const MaskDef &def, const P origin, const Tile<Coord2D> &tile,
                const double margin) {
  const auto mask =
      construct_mask<Coord2D>(def.shape, {origin.x, origin.y}, def.params, {});
  return convex_sets_intersect(
      [&](const Coord2D &u) { return mask_support(mask, u); },
      [&](const Coord2D &u) { return tile.support(u); },
      mask.bounding_disc_.origin_ - tile.c_radius_.origin_, margin);
}

// Touching sets and gaps just below and above the margin delta
void test_touching() {
  const double delta = EXACT_FILTER_REL_TOL * 4;
  const auto a = rectangle(0.5, 0.5, {1.}); // [0, 1]^2
  require(tile_pair_meets(a, rectangle(1.5, 0.5, {1.}), 0.), "shared edge");
  require(tile_pair_meets(a, rectangle(1.5, 1.5, {1.}), 0.), "shared vertex");
  require(tile_pair_meets(a, a, 0.), "A = A");
  for (const double side : {-1., 1.}) {
    const double gap = delta * (1 + side * 1e-3);
    const bool expected = side < 0;
    require(tile_pair_meets(a, rectangle(1.5 + gap, 0.5, {1.}), delta) ==
                expected,
            "edge gap at the margin");
    require(tile_pair_meets(a, rectangle(1.5 + gap, 1.5 + gap, {1.}),
                            delta * std::numbers::sqrt2) == expected,
            "vertex gap at the margin");
  }

  // Curved and polygonal masks tangent to the top edge y = 1 of a wide tile,
  // their lowest point placed analytically
  const auto wide = rectangle(0., 0.5, {10., 1.});
  for (int degrees = 0; degrees < 180; degrees += 15) {
    const double t = degrees * std::numbers::pi / 180;
    const double a_axis = 2., b_axis = 0.3;
    const double depth = std::sqrt(a_axis * a_axis * std::sin(t) * std::sin(t) +
                                   b_axis * b_axis * std::cos(t) * std::cos(t));
    const MaskDef ellipse{MASK_SHAPE::ELLIPTICAL,
                          {a_axis, b_axis, double(degrees)}};
    require(mask_meets(ellipse, {0.3, 1 + depth}, wide, delta),
            "tangent ellipse");
    require(!mask_meets(ellipse, {0.3, 1 + depth + 4 * delta}, wide, delta),
            "ellipse past the margin");
  }
  const MaskDef circle{MASK_SHAPE::CIRCULAR, {0.5}};
  require(mask_meets(circle, {0.3, 1.5}, wide, delta), "tangent circle");
  require(!mask_meets(circle, {0.3, 1.5 + 4 * delta}, wide, delta),
          "circle past the margin");
  const MaskDef corner{MASK_SHAPE::TRIANGULAR, {1., 1., -1., 1.}};
  require(mask_meets(corner, {0.3, 1.}, wide, 0.), "triangle anchor on edge");
  require(!mask_meets(corner, {0.3, 1 + 4 * delta}, wide, delta),
          "triangle anchor past the margin");
}

// Every uncertain path reports an intersection
void test_uncertain_paths() {
  const HullSupport left{
      {construct_coord_2D(-3, -1), construct_coord_2D(-2, 1)}};
  const HullSupport right{
      {construct_coord_2D(2, -1), construct_coord_2D(3, 1)}};
  require(!convex_sets_intersect(left, right, construct_coord_2D(-1, 0), 0.),
          "separated segments");
  require(convex_sets_intersect(left, right, Coord2D(), 0.), "zero direction");

  const auto nan_support = [](const Coord2D &) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    return construct_coord_2D(nan, nan);
  };
  require(
      convex_sets_intersect(nan_support, right, construct_coord_2D(1, 0), 0.),
      "NaN support");
  require(
      convex_sets_intersect(left, nan_support, construct_coord_2D(1, 0), 0.),
      "NaN support of B");

  // The segment y = 1 seen from the origin needs two iterations to separate
  // when searched along x first
  const HullSupport segment{
      {construct_coord_2D(-10, 1), construct_coord_2D(10, 1)}};
  const HullSupport origin{{Coord2D()}};
  require(!convex_sets_intersect(segment, origin, construct_coord_2D(1, 0), 0.),
          "separated segment");
  require(
      convex_sets_intersect(segment, origin, construct_coord_2D(1, 0), 0., 1),
      "iteration cap");

  // Degenerate sets: points and segments through the origin
  require(convex_sets_intersect(origin, origin, construct_coord_2D(1, 0), 0.),
          "equal points");
  require(!convex_sets_intersect(HullSupport{{construct_coord_2D(1e-3, 0)}},
                                 origin, construct_coord_2D(1, 0), 0.),
          "distinct points");
  const HullSupport through{
      {construct_coord_2D(-2, -2), construct_coord_2D(2, 2)}};
  require(convex_sets_intersect(through, origin, construct_coord_2D(1, -1), 0.),
          "segment through the origin");
  require(
      convex_sets_intersect(through, segment, construct_coord_2D(1, -1), 0.),
      "crossing segments");
}

struct Sphere {
  Coord3D centre;
  double radius;
  Coord3D operator()(const Coord3D &u) const {
    const double n = vector_norm(u);
    return 0 < n ? centre + u * (radius / n) : centre;
  }
};

struct Box {
  Coord3D centre, half;
  Coord3D operator()(const Coord3D &u) const {
    return centre + construct_coord_3D(std::signbit(u.x_) ? -half.x_ : half.x_,
                                       std::signbit(u.y_) ? -half.y_ : half.y_,
                                       std::signbit(u.z_) ? -half.z_ : half.z_);
  }
};

double box_gap(const Box &a, const Box &b) {
  const Coord3D d = a.centre - b.centre;
  const double g[3] = {std::fabs(d.x_) - a.half.x_ - b.half.x_,
                       std::fabs(d.y_) - a.half.y_ - b.half.y_,
                       std::fabs(d.z_) - a.half.z_ - b.half.z_};
  double gap2 = 0, depth = -std::numeric_limits<double>::infinity();
  for (const double x : g) {
    gap2 += x > 0 ? x * x : 0;
    depth = std::max(depth, x);
  }
  return depth <= 0 ? depth : std::sqrt(gap2);
}

double sphere_box_gap(const Sphere &s, const Box &b) {
  const Coord3D d = s.centre - b.centre;
  const Coord3D clamped =
      construct_coord_3D(std::clamp(d.x_, -b.half.x_, b.half.x_),
                         std::clamp(d.y_, -b.half.y_, b.half.y_),
                         std::clamp(d.z_, -b.half.z_, b.half.z_));
  return vector_norm(d - clamped) - s.radius;
}

// The generic query instantiated for 3D with sphere and box supports, against
// their exact distances
void test_3d() {
  Rng rng(2026);
  const auto point = [&](const double reach) {
    return construct_coord_3D(uniform(rng, -reach, reach),
                              uniform(rng, -reach, reach),
                              uniform(rng, -reach, reach));
  };
  const auto positive = [&] {
    return construct_coord_3D(uniform(rng, 0.05, 1.), uniform(rng, 0.05, 1.),
                              uniform(rng, 0.05, 1.));
  };
  long intersecting = 0, separated = 0;
  const auto judge = [&](const bool got, const double gap, const double margin,
                         const std::string &what) {
    if (gap <= margin - 1e-9) {
      require(got, "3D GJK misses an intersection: " + what);
      ++intersecting;
    } else if (margin + 1e-9 < gap) {
      require(!got, "3D GJK reports a gap as an intersection: " + what);
      ++separated;
    }
  };

  for (int k = 0; k < 5000; ++k) {
    const double margin = k % 2 ? 0. : 0.1;
    const Sphere s1{point(1.), uniform(rng, 0.1, 1.)};
    const Sphere s2{point(1.5), uniform(rng, 0.1, 1.)};
    judge(convex_sets_intersect(s1, s2, s1.centre - s2.centre, margin),
          vector_norm(s1.centre - s2.centre) - s1.radius - s2.radius, margin,
          "spheres");

    const Box b1{point(1.), positive()};
    const Box b2{point(1.5), positive()};
    judge(convex_sets_intersect(b1, b2, b1.centre - b2.centre, margin),
          box_gap(b1, b2), margin, "boxes");

    judge(convex_sets_intersect(s2, b1, s2.centre - b1.centre, margin),
          sphere_box_gap(s2, b1), margin, "sphere and box");
  }
  require(intersecting > 1000 && separated > 1000,
          "one-sided 3D cases: " + std::to_string(intersecting) + " " +
              std::to_string(separated));

  const Box unit{Coord3D(), construct_coord_3D(0.5, 0.5, 0.5)};
  require(convex_sets_intersect(unit,
                                Box{construct_coord_3D(1, 1, 1), unit.half},
                                construct_coord_3D(-1, -1, -1), 0.),
          "boxes sharing a vertex");
  require(convex_sets_intersect(Sphere{construct_coord_3D(0, 0, 1), 0.5}, unit,
                                construct_coord_3D(0, 0, 1), 1e-9),
          "sphere tangent to a box face");
}
} // namespace

int main() {
  std::deque<Tile<Coord2D>> roots;
  const auto catalog = tile_catalog(roots, 2);

  test_tile_support(catalog);
  test_mask_support();
  test_simplex();
  test_polygons();
  test_masks_and_tiles(catalog);
  test_touching();
  test_uncertain_paths();
  test_3d();
  return 0;
}
