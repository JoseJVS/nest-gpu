/*
 *  blueprint_range_test.cpp
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

// The blueprint range tests never reject a tile pair (or a point and a tile)
// holding a node pair in the blueprint, with its offset, for every blueprint
// shape. For circular blueprints they are exact: true only if the polygon
// distance is within the radius. The reference for a tile pair is the exact
// set of displacements between them, D = B - A - offset.

#include <cassert>
#include <random>

#include "bounding_box.h"
#include "coordinate_geometry.h"
#include "creator_registry.h"
#include "geometry_test_helpers.h"
#include "gf2d_geometry.h"
#include "gf_collection.h"
#include "gf_creators.h"
#include "grid_generation.h"
#include "mask.h"
#include "test_helpers.h"
#include "thread_aligned_array.h"
#include "tile.h"
#include "type_erasure_helpers.h"
#include "vp_interface.h"

using namespace sapi;
using namespace geometry_test;

struct BlueprintDef {
  MASK_SHAPE shape;
  std::vector<space_t> params;
};

const std::vector<BlueprintDef> BLUEPRINTS{
    {MASK_SHAPE::CIRCULAR, {0.3}},
    {MASK_SHAPE::CIRCULAR, {1.2}},
    {MASK_SHAPE::ELLIPTICAL, {2., 0.2}},
    {MASK_SHAPE::ELLIPTICAL, {2., 0.2, 45.}},
    {MASK_SHAPE::PARALLELOGRAM, {0.8, 0., 0.2, 0.4}},
    {MASK_SHAPE::TRIANGULAR, {0.9, 0., 0., 0.9}},
    {MASK_SHAPE::ANNULAR, {0.3, 0.9}},
    {MASK_SHAPE::ANNULAR, {1.5, 2.2}},
    {MASK_SHAPE::ANNULAR, {0., 1.2}}};

// Whether the blueprint holds for some displacement in the polygon region
bool region_meets_blueprint(const Mask<Coord2D> &blueprint,
                            const std::vector<P> &displacements) {
  const Coord2D zero = construct_coord_2D(0., 0.);
  for (const auto &d : sample_region(displacements, 30, 300))
    if (blueprint.coord_in_mask(zero, d + blueprint.anchor_).first)
      return true;
  return false;
}

// Center of the blueprint's bounding disk in the frame of the displacement
// regions (relative to the anchor): zero for the centered shapes
P disk_center(const Mask<Coord2D> &blueprint) {
  return from(blueprint.bounding_disk_.origin_) - from(blueprint.anchor_);
}

// distance is measured from the bounding disk center
void check_verdict(const bool verdict, const double distance,
                   const Mask<Coord2D> &blueprint, const BlueprintDef &def,
                   const std::vector<P> &region) {
  const double radius = std::sqrt(blueprint.bounding_disk_.radius2_);
  if (def.shape == MASK_SHAPE::CIRCULAR) {
    if (verdict)
      assert(distance <= radius * (1 + 1e-5) + 1e-6);
    else
      assert(distance >= radius * (1 - 1e-5) - 1e-6);
  } else if (def.shape == MASK_SHAPE::ANNULAR) {
    // Exact up to the margins: the region meets the outer disk and is not
    // inside the inner one. Its farthest point from the center is a vertex.
    const P center = disk_center(blueprint);
    double farthest = 0.;
    for (const auto &v : region)
      farthest =
          std::fmax(farthest, std::hypot(v.x - center.x, v.y - center.y));
    const double inner = def.params[0];
    if (verdict) {
      assert(distance <= radius * (1 + 1e-5) + 1e-6);
      assert(farthest >= inner * (1 - 1e-5) - 1e-6);
    } else {
      assert(distance >= radius * (1 - 1e-5) - 1e-6 ||
             farthest <= inner * (1 + 1e-5) + 1e-6);
      if (distance <= radius)
        assert(!region_meets_blueprint(blueprint, region));
    }
  } else if (!verdict && distance <= radius) {
    assert(!region_meets_blueprint(blueprint, region));
  }
}

void check_pair(const Tile<Coord2D> &a, const Tile<Coord2D> &b,
                const Mask<Coord2D> &blueprint, const BlueprintDef &def) {
  const auto d = minkowski_difference(a, b, from(blueprint.anchor_));
  check_verdict(blueprint.tiles_within_mask_range(a, b),
                region_distance(disk_center(blueprint), d), blueprint, def, d);
}

// A point at coord drives: pool points b in tile B with b - coord - anchor in
// the blueprint
void check_point(const Coord2D &coord, const Tile<Coord2D> &b,
                 const Mask<Coord2D> &blueprint, const BlueprintDef &def) {
  std::vector<P> d;
  for (const auto &v : polygon(b))
    d.push_back(v - from(coord) - from(blueprint.anchor_));
  check_verdict(blueprint.overlap_with_tile_surface(coord, b),
                region_distance(disk_center(blueprint), d), blueprint, def, d);
}

void test_grids() {
  TAArray<GFCollection<Coord2D>> gc_array;
  CreatorRegistry<GridTargetPositionShifts<Coord2D>> gscr;
  CreatorRegistry<ShiftedOriginCreator<Coord2D>> socr;
  CreatorRegistry<CachedTileCreator<Coord2D>> ctcr;
  initialize_gsc_registry(gscr);
  initialize_soc_registry(socr);
  initialize_ctc_registry(ctcr);

  std::mt19937_64 rng(2026);
  std::uniform_real_distribution<double> unit(-1, 1);
  std::vector<P> offsets{{0, 0}, {1, 0}, {2, 0}, {0, 0.5}};
  for (int i = 0; i < 20; ++i)
    offsets.push_back(P{unit(rng), unit(rng)} * 3.);

  struct GridDef {
    std::string shape;
    std::vector<tileidx_t> dims;
    std::vector<space_t> sides;
  };
  const std::vector<GridDef> grids{{"rectangle", {4, 3}, {1.}},
                                   {"rectangle", {3, 4}, {2., 1.}},
                                   {"triangle", {4, 4}, {1.}},
                                   {"hexagon", {4, 4}, {1.}}};

  for (const auto &g : grids) {
    gc_array.clone(construct_gf_collection<Coord2D>(
        {0., 0.}, g.dims, g.shape, g.sides, {0}, gscr, socr, ctcr));
    auto grid = generate_tile_grid(gc_array);
    for (auto &pos : grid.positions_)
      pos.tile_.split(2, false);

    std::vector<const Tile<Coord2D> *> level;
    for (const auto &pos : grid.positions_)
      level.push_back(&pos.tile_);

    for (int l = 0; l <= 2; ++l) {
      // Every pair of whole tiles; a random subset of sub-tile pairs
      std::vector<std::pair<std::size_t, std::size_t>> pairs;
      if (l == 0) {
        for (std::size_t i = 0; i < level.size(); ++i)
          for (std::size_t j = 0; j < level.size(); ++j)
            pairs.emplace_back(i, j);
      } else {
        for (int k = 0; k < 60; ++k)
          pairs.emplace_back(rng() % level.size(), rng() % level.size());
      }

      for (const auto &def : BLUEPRINTS)
        for (const auto &offset : offsets) {
          const auto blueprint = construct_mask<Coord2D>(
              def.shape, {}, def.params, {offset.x, offset.y});
          for (const auto &[i, j] : pairs)
            check_pair(*level[i], *level[j], blueprint, def);

          const double radius = std::sqrt(blueprint.bounding_disk_.radius2_);
          for (std::size_t i = 0; i < level.size(); i += 1 + level.size() / 8) {
            const auto &b = *level[i];
            const double reach = std::sqrt(b.c_radius_.radius2_) + radius;
            for (int k = 0; k < 4; ++k)
              check_point(to_coord(from(b.c_radius_.origin_) -
                                   from(blueprint.bounding_disk_.origin_) +
                                   P{unit(rng), unit(rng)} * (1.2 * reach)),
                          b, blueprint, def);
          }
        }

      std::vector<const Tile<Coord2D> *> next;
      for (const auto *t : level)
        for (const auto &st : t->sub_tiles_)
          next.push_back(&st);
      level = next;
    }
  }
}

// Unit squares A, B (right of A), C (two to the right) and D (above B):
// offsets and shapes for which the previous closest-pair test dropped pairs
void test_regressions() {
  const auto square = [](const double x, const double y) {
    return Tile<Coord2D>(TILE_SHAPE::RECTANGLE, construct_coord_2D(x, y), {1.},
                         {});
  };
  const auto a = square(0.5, 0.5), b = square(1.5, 0.5), c = square(2.5, 0.5),
             d = square(1.5, 1.5);
  const auto circle = [](const double r, const double x, const double y) {
    return construct_mask<Coord2D>(MASK_SHAPE::CIRCULAR, {}, {r}, {x, y});
  };
  const auto ellipse45 =
      construct_mask<Coord2D>(MASK_SHAPE::ELLIPTICAL, {}, {2., 0.2, 45.}, {});

  assert(circle(0.3, 0., 0.).tiles_within_mask_range(a, a));
  assert(circle(0.3, 0., 0.).tiles_within_mask_range(a, b));
  assert(circle(1.2, 0., 0.).tiles_within_mask_range(a, c));
  assert(circle(0.3, 0., 0.5).tiles_within_mask_range(a, b));
  assert(circle(0.3, 1., 0.).tiles_within_mask_range(a, b));
  assert(circle(0.3, 2., 0.).tiles_within_mask_range(a, c));
  assert(!circle(0.3, 1., 0.).tiles_within_mask_range(b, a));
  assert(ellipse45.tiles_within_mask_range(a, c));
  assert(ellipse45.tiles_within_mask_range(a, d));
  assert(circle(0.21, -0.2, -0.9).tiles_within_mask_range(a, b));
}

// Rings: displacements all in the hole, a pair that straddles the ring, a small
// ring inside a big tile, and r_in = 0 against the circle
void test_annulus() {
  const auto square = [](const double x, const double y, const double side) {
    return Tile<Coord2D>(TILE_SHAPE::RECTANGLE, construct_coord_2D(x, y),
                         {side}, {});
  };
  const auto a = square(0.5, 0.5, 1.), b = square(1.5, 0.5, 1.),
             c = square(2.5, 0.5, 1.);
  const auto ring = [](const double r_in, const double r_out, const double x,
                       const double y) {
    return construct_mask<Coord2D>(MASK_SHAPE::ANNULAR, {}, {r_in, r_out},
                                   {x, y});
  };

  // Displacements of (a, a) lie in [-1, 1]^2, within 1.42 of the center
  assert(!ring(1.5, 2., 0., 0.).tiles_within_mask_range(a, a));
  assert(ring(1., 2., 0., 0.).tiles_within_mask_range(a, a));
  assert(ring(0.3, 0.9, 0., 0.).tiles_within_mask_range(a, a));
  // (a, c): x in [1, 3]; the ring at distance 2 is crossed, 0.5 is too short,
  // and with the offset (2, 0) the displacements are in [-1, 1]^2 again
  assert(ring(1.5, 2.5, 0., 0.).tiles_within_mask_range(a, c));
  assert(!ring(0.2, 0.5, 0., 0.).tiles_within_mask_range(a, c));
  assert(!ring(1.5, 2., 2., 0.).tiles_within_mask_range(a, c));
  assert(ring(0.3, 0.9, 2., 0.).tiles_within_mask_range(a, c));

  // Point at (0.5, 0.5) against b: displacements x in [0.5, 1.5], y in
  // [-0.5, 0.5]
  const Coord2D p = construct_coord_2D(0.5, 0.5);
  assert(!ring(2., 3., 0., 0.).overlap_with_tile_surface(p, b));
  assert(ring(1., 1.2, 0., 0.).overlap_with_tile_surface(p, b));
  assert(ring(0.2, 0.6, 0., 0.).overlap_with_tile_surface(p, b));
  assert(!ring(0.2, 0.4, 0., 0.).overlap_with_tile_surface(p, b));

  // A small ring inside a big tile meets it; a small tile in the hole does not
  const auto big = square(0., 0., 10.);
  const auto small = square(0.1, 0., 0.2);
  assert(ring(0.1, 0.2, 1., 1.).tiles_within_mask_range(big, big));
  assert(ring(0.1, 0.2, 0., 0.).tiles_within_mask_range(big, big));
  assert(ring(0.1, 0.2, 0., 0.)
             .overlap_with_tile_surface(construct_coord_2D(1., 1.), big));
  assert(!ring(1., 2., 0., 0.).tiles_within_mask_range(small, small));
  assert(!ring(1., 2., 0., 0.)
              .overlap_with_tile_surface(construct_coord_2D(0., 0.), small));

  // r_in = 0 is the circle
  for (const auto *x : {&a, &b, &c, &big, &small})
    for (const auto *y : {&a, &b, &c, &big, &small})
      for (const P &o : {P{0., 0.}, P{1., 0.}, P{-0.5, 2.}}) {
        const auto disk = ring(0., 1.2, o.x, o.y);
        const auto circle = construct_mask<Coord2D>(MASK_SHAPE::CIRCULAR, {},
                                                    {1.2}, {o.x, o.y});
        assert(disk.tiles_within_mask_range(*x, *y) ==
               circle.tiles_within_mask_range(*x, *y));
        assert(disk.overlap_with_tile_surface(x->c_radius_.origin_, *y) ==
               circle.overlap_with_tile_surface(x->c_radius_.origin_, *y));
      }
}

int main() {
  init_omp();
  test_grids();
  test_regressions();
  test_annulus();
  return 0;
}
