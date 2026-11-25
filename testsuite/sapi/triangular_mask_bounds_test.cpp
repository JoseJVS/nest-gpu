/*
 *  triangular_mask_bounds_test.cpp
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

// Bounding disks of the masks (plan Q7.16a5). Every mask shape's bounding
// disk contains the whole shape and is centered on a point of the shape. For
// triangular masks the disk is the smallest enclosing disk (circumcenter, or
// the longest-edge midpoint for obtuse triangles), no larger than the old
// disk centered on the anchor vertex, and the node-level tests give the same
// verdicts and the same displacements as with the old disk. The tile filters
// never reject a tile (or tile pair) holding a point (or node pair) in the
// mask.

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <random>
#include <string>
#include <vector>

#include "coordinate_geometry.h"
#include "geometry_test_helpers.h"
#include "mask.h"
#include "test_helpers.h"
#include "tile.h"
#include "vp_interface.h"

using namespace sapi;
using namespace geometry_test;

namespace {
// Prints what failed and aborts; independent of NDEBUG
void require(const bool condition, const std::string &what) {
  if (!condition) {
    std::fprintf(stderr, "triangular_mask_bounds_test: %s\n", what.c_str());
    std::abort();
  }
}

struct MaskDef {
  std::string name;
  MASK_SHAPE shape;
  std::vector<space_t> params;
};

// Triangles {0, a0, a1}: acute, right, obtuse, slivers, both orientations,
// tiny and large; the centered shapes for the containment checks
std::vector<MaskDef> triangle_defs() {
  return {{"acute", MASK_SHAPE::TRIANGULAR, {1., 0., 0.4, 0.9}},
          {"acute_cw", MASK_SHAPE::TRIANGULAR, {0.4, 0.9, 1., 0.}},
          {"right", MASK_SHAPE::TRIANGULAR, {1.2, 0., 0., 1.2}},
          {"obtuse_anchor", MASK_SHAPE::TRIANGULAR, {1., 0., -0.8, 0.3}},
          {"obtuse_far", MASK_SHAPE::TRIANGULAR, {1., 0., 1.6, 0.2}},
          {"sliver", MASK_SHAPE::TRIANGULAR, {0.25, 0.02, 0.25, -0.02}},
          {"sliver_long", MASK_SHAPE::TRIANGULAR, {2., 0.01, 1.9, -0.012}},
          {"sliver_extreme", MASK_SHAPE::TRIANGULAR, {1., 0., 0.5, 1e-9}},
          // Obtuse sliver whose center check needs an edge-length tolerance
          // (an area-relative one rejects it)
          {"sliver_rounding",
           MASK_SHAPE::TRIANGULAR,
           {-72.473701269246334, 463.33538390592724, 9.8376691017500466,
            -62.893713342376799}},
          {"tiny", MASK_SHAPE::TRIANGULAR, {1e-3, 2e-4, 3e-4, 9e-4}},
          {"large", MASK_SHAPE::TRIANGULAR, {800., 100., 200., 900.}}};
}

std::vector<MaskDef> centered_defs() {
  return {{"circle", MASK_SHAPE::CIRCULAR, {0.35}},
          {"ellipse", MASK_SHAPE::ELLIPTICAL, {2., 0.2, 30.}},
          {"parallelogram", MASK_SHAPE::PARALLELOGRAM, {1., 0.2, -0.3, 0.8}}};
}

// Polygon of a triangular or parallelogram mask in absolute coordinates
std::vector<P> mask_polygon(const MaskDef &def, const P anchor) {
  const P a0{def.params[0], def.params[1]}, a1{def.params[2], def.params[3]};
  std::vector<P> v =
      def.shape == MASK_SHAPE::TRIANGULAR
          ? std::vector<P>{anchor, anchor + a0, anchor + a1}
          : std::vector<P>{anchor + a0, anchor + a1, anchor - a0, anchor - a1};
  if (orientation(v) < 0)
    std::reverse(v.begin(), v.end());
  return v;
}

// Ellipse boundary and interior samples (semi-axes a, b, rotation in degrees)
std::vector<Coord2D> ellipse_samples(const MaskDef &def, const P anchor) {
  const double a = def.params[0], b = def.params[1],
               r = def.params[2] * M_PI / 180.;
  std::vector<Coord2D> s;
  for (int k = 0; k < 720; ++k)
    for (const double f : {1., 0.6, 0.2}) {
      const double t = 2 * M_PI * k / 720.;
      const P e{f * a * std::cos(t), f * b * std::sin(t)};
      s.push_back(to_coord(anchor + P{e.x * std::cos(r) - e.y * std::sin(r),
                                      e.x * std::sin(r) + e.y * std::cos(r)}));
    }
  return s;
}

std::vector<Coord2D> shape_samples(const MaskDef &def, const P anchor) {
  switch (def.shape) {
  case MASK_SHAPE::CIRCULAR: {
    std::vector<Coord2D> s;
    for (int k = 0; k < 720; ++k)
      for (const double f : {1., 0.5}) {
        const double t = 2 * M_PI * k / 720.;
        s.push_back(to_coord(anchor + P{f * def.params[0] * std::cos(t),
                                        f * def.params[0] * std::sin(t)}));
      }
    return s;
  }
  case MASK_SHAPE::ELLIPTICAL:
    return ellipse_samples(def, anchor);
  default:
    return sample_region(mask_polygon(def, anchor), 60, 900);
  }
}

// Radius of the smallest disk enclosing the triangle {0, a0, a1}, computed
// independently of SAPI: half the longest edge when the triangle is obtuse or
// right, the circumradius abc / (4 area) otherwise
double minimal_enclosing_radius(const P a0, const P a1) {
  const P v[3] = {{0., 0.}, a0, a1};
  const double e[3] = {norm(v[1] - v[2]), norm(v[2] - v[0]), norm(v[0] - v[1])};
  for (int i = 0; i < 3; ++i)
    if (dot(v[(i + 1) % 3] - v[i], v[(i + 2) % 3] - v[i]) <= 0)
      return 0.5 * e[i];
  return e[0] * e[1] * e[2] / (2. * std::fabs(cross(a0, a1)));
}

bool same_bits(const Coord2D &a, const Coord2D &b) {
  return std::bit_cast<uint64_t>(a.x_) == std::bit_cast<uint64_t>(b.x_) &&
         std::bit_cast<uint64_t>(a.y_) == std::bit_cast<uint64_t>(b.y_);
}

// The old triangular test: disk centered on the anchor with radius
// max(|a0|, |a1|), then the same exact test with the mask's stored vectors and
// determinant (as computed by the library), so only the prefilter differs
OptDisp<Coord2D> old_triangular(const Coord2D &d, const Mask<Coord2D> &mask) {
  const Coord2D &a0 = mask.helper_vectors_[0];
  const Coord2D &a1 = mask.helper_vectors_[1];
  OptDisp<Coord2D> od(false, construct_displacement(Coord2D(d)));
  if (leq_test(od.second.distance2_,
               std::fmax(vector_norm2(a0), vector_norm2(a1))))
    od.first = algebraic_projection_comparison<true>(
        od.second.displacement_, a0, a1, mask.helper_scalars_[0]);
  return od;
}

// 1. The bounding disk contains the shape and is centered on it
void test_disk_contains_shape() {
  std::vector<MaskDef> defs = triangle_defs();
  for (const auto &def : centered_defs())
    defs.push_back(def);

  const P anchor{0.7, -1.3};
  for (const auto &def : defs) {
    const auto mask = construct_mask<Coord2D>(def.shape, {anchor.x, anchor.y},
                                              def.params, {});
    const auto &disk = mask.bounding_disk_;

    for (const auto &s : shape_samples(def, anchor))
      require(disk.coord_in_radius(s).first,
              "disk does not contain " + def.name + " at " + s.to_string());

    require(mask.coord_in_mask(disk.origin_).first,
            "disk center outside the " + def.name + " mask");
  }
}

// 2. Triangles: the smallest enclosing disk, never larger than the old one
void test_triangle_radius() {
  for (const auto &def : triangle_defs()) {
    const P a0{def.params[0], def.params[1]}, a1{def.params[2], def.params[3]};
    const auto mask =
        construct_mask<Coord2D>(def.shape, {0., 0.}, def.params, {});
    const double radius = std::sqrt(mask.bounding_disk_.radius2_);
    const double expected = minimal_enclosing_radius(a0, a1);
    const double old_radius = std::max(norm(a0), norm(a1));

    require(std::fabs(radius - expected) <= 1e-9 * expected,
            def.name + ": radius " + std::to_string(radius) +
                " is not the minimal enclosing radius " +
                std::to_string(expected));
    require(radius <= old_radius * (1 + 1e-9),
            def.name + ": radius larger than the old disk");
  }
}

// 3. Node-level tests: same verdicts and bit-identical displacements as the
// old disk, inside, on edges and vertices, and around the triangle
void test_node_tests_unchanged() {
  std::mt19937_64 rng(2026);
  const P anchor{-0.4, 2.1}, offset{0.3, -0.2};
  for (const auto &def : triangle_defs()) {
    const Coord2D a0 = construct_coord_2D(def.params[0], def.params[1]);
    const Coord2D a1 = construct_coord_2D(def.params[2], def.params[3]);
    const auto mask = construct_mask<Coord2D>(def.shape, {anchor.x, anchor.y},
                                              def.params, {offset.x, offset.y});
    const auto blueprint = construct_mask<Coord2D>(def.shape, {}, def.params,
                                                   {offset.x, offset.y});

    // Points of the triangle (anchor frame of the source/target mask is
    // anchor + offset) and a box around it
    const P origin = anchor + offset;
    std::vector<Coord2D> points = shape_samples(def, origin);
    const double extent = std::max(norm(from(a0)), norm(from(a1)));
    std::uniform_real_distribution<double> box(-2 * extent, 2 * extent);
    for (int k = 0; k < 4000; ++k)
      points.push_back(to_coord(origin + P{box(rng), box(rng)}));

    for (const auto &c : points) {
      const auto now = mask.coord_in_mask(c);
      const auto old = old_triangular(c - mask.anchor_, mask);
      if (now.first != old.first ||
          !same_bits(now.second.displacement_, old.second.displacement_))
        require(false,
                def.name + ": one-argument test changed at " + c.to_string());

      // Blueprint form: a driver at the anchor, the pool node at c
      const Coord2D a = to_coord(anchor);
      const auto now2 = blueprint.coord_in_mask(a, c);
      const auto old2 = old_triangular(c - (a + blueprint.anchor_), blueprint);
      if (now2.first != old2.first ||
          !same_bits(now2.second.displacement_, old2.second.displacement_))
        require(false,
                def.name + ": two-argument test changed at " + c.to_string());
    }
  }
}

// 4. Tile filters never reject a tile or tile pair holding a point or node
// pair in the triangle
void test_tile_filters_sound() {
  std::deque<Tile<Coord2D>> roots;
  const auto catalog = tile_catalog(roots, 1);
  std::mt19937_64 rng(7);
  std::uniform_real_distribution<double> around(-2.5, 2.5);

  for (const auto &def : triangle_defs()) {
    if (def.name == "tiny" || def.name == "large")
      continue; // scale-specific cases covered by tests 1-3
    const auto blueprint =
        construct_mask<Coord2D>(def.shape, {}, def.params, {0.2, -0.1});

    for (const auto &named : catalog) {
      const Tile<Coord2D> &tile = *named.tile;
      const auto tile_points = sample_region(polygon(tile), 12, 60);

      // Source/target mask placed at random around the tile
      for (int k = 0; k < 20; ++k) {
        const P c = from(tile.c_radius_.origin_) + P{around(rng), around(rng)};
        const auto mask =
            construct_mask<Coord2D>(def.shape, {c.x, c.y}, def.params, {});
        if (mask.tile_overlap(tile) == OVERLAP_LEVEL::NONE &&
            std::any_of(
                tile_points.begin(), tile_points.end(),
                [&](const Coord2D &p) { return mask.coord_in_mask(p).first; }))
          require(false, def.name + ": tile_overlap rejected " + named.name);
      }

      // Blueprint: tile pairs and point versus tile
      for (const auto &other : catalog) {
        const Tile<Coord2D> &pool = *other.tile;
        if (blueprint.tiles_within_mask_range(tile, pool))
          continue;
        const auto pool_points = sample_region(polygon(pool), 8, 40);
        for (const auto &pa : sample_region(polygon(tile), 8, 40))
          for (const auto &pb : pool_points)
            require(!blueprint.coord_in_mask(pa, pb).first,
                    def.name + ": tiles_within_mask_range rejected " +
                        named.name + " -> " + other.name);
      }

      for (int k = 0; k < 10; ++k) {
        const Coord2D driver = to_coord(from(tile.c_radius_.origin_) +
                                        P{around(rng), around(rng)});
        if (!blueprint.overlap_with_tile_surface(driver, tile) &&
            std::any_of(tile_points.begin(), tile_points.end(),
                        [&](const Coord2D &p) {
                          return blueprint.coord_in_mask(driver, p).first;
                        }))
          require(false, def.name + ": overlap_with_tile_surface rejected " +
                             named.name);
      }
    }
  }
}
} // namespace

int main() {
  init_omp();
  test_disk_contains_shape();
  test_triangle_radius();
  test_node_tests_unchanged();
  test_tile_filters_sound();
  return 0;
}
