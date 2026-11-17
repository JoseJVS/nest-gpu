/*
 *  mask_tile_overlap_test.cpp
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

// Mask::tile_overlap never reports NONE for a tile that holds a point of the
// mask, never reports FULL for a tile with a point outside it, and for
// circular masks only reports PARTIAL for tiles within the radius.

#include <random>

#include "coordinate_geometry.h"
#include "geometry_test_helpers.h"
#include "mask.h"
#include "test_helpers.h"
#include "tile.h"

using namespace sapi;
using namespace geometry_test;

struct MaskDef {
  MASK_SHAPE shape;
  std::vector<space_t> params;
};

const std::vector<MaskDef> MASKS{
    {MASK_SHAPE::CIRCULAR, {0.35}},
    {MASK_SHAPE::CIRCULAR, {1.5}},
    {MASK_SHAPE::ELLIPTICAL, {2., 0.2}},
    {MASK_SHAPE::ELLIPTICAL, {2., 0.2, 45.}},
    {MASK_SHAPE::PARALLELOGRAM, {1., 0.2, -0.3, 0.8}},
    {MASK_SHAPE::TRIANGULAR, {1.2, 0., 0., 1.2}}};

// Outside with a margin: moving 1e-5 of the way to the mask origin keeps it
// outside (all mask shapes are convex and contain their origin)
bool robustly_outside(const Mask<Coord2D> &mask, const Coord2D &coord) {
  const P o = from(mask.c_radius_.origin_);
  const P closer = o + (from(coord) - o) * (1 - 1e-5);
  return !mask.coord_in_mask(coord).first &&
         !mask.coord_in_mask(to_coord(closer)).first;
}

void check_overlap(const Mask<Coord2D> &mask, const MaskDef &def,
                   const Tile<Coord2D> &tile,
                   const std::vector<Coord2D> &samples) {
  const auto in_mask = [&](const Coord2D &c) {
    return mask.coord_in_mask(c).first;
  };

  switch (mask.tile_overlap(tile)) {
  case OVERLAP_LEVEL::NONE:
    assert(std::none_of(samples.begin(), samples.end(), in_mask));
    break;

  case OVERLAP_LEVEL::FULL:
    assert(std::none_of(samples.begin(), samples.end(), [&](const auto &c) {
      return robustly_outside(mask, c);
    }));
    break;

  default:
    if (def.shape == MASK_SHAPE::CIRCULAR)
      assert(region_distance(from(mask.c_radius_.origin_), polygon(tile)) <=
             def.params[0] * (1 + 1e-5) + 1e-6);
    break;
  }
}

// Every mask placed at random around every tile of the catalog
void test_catalog() {
  std::deque<Tile<Coord2D>> roots;
  const auto catalog = tile_catalog(roots, 2);
  std::mt19937_64 rng(2026);
  std::uniform_real_distribution<double> unit(-1, 1);

  for (const auto &nt : catalog) {
    const auto samples = sample_region(polygon(*nt.tile), 40, 300);
    const double tile_radius = std::sqrt(nt.tile->c_radius_.radius2_);
    for (const auto &def : MASKS) {
      const double mask_radius =
          std::sqrt(construct_mask<Coord2D>(def.shape, {0., 0.}, def.params, {})
                        .c_radius_.radius2_);
      for (int k = 0; k < 40; ++k) {
        const P c =
            from(nt.tile->c_radius_.origin_) +
            P{unit(rng), unit(rng)} * ((tile_radius + mask_radius) * 1.1);
        check_overlap(
            construct_mask<Coord2D>(def.shape, {c.x, c.y}, def.params, {}), def,
            *nt.tile, samples);
      }
    }
  }
}

// A non-circular mask whose closest perimeter point is outside it, and a
// circle next to a 2x1 rectangle whose two nearest vertices span the wrong edge
void test_regressions() {
  const Tile<Coord2D> square(TILE_SHAPE::RECTANGLE,
                             construct_coord_2D(1.1, 1.5), {1.}, {});
  const auto ellipse45 = construct_mask<Coord2D>(MASK_SHAPE::ELLIPTICAL,
                                                 {0., 0.}, {2., 0.2, 45.}, {});
  assert(ellipse45.tile_overlap(square) == OVERLAP_LEVEL::PARTIAL);

  const auto ellipse0 =
      construct_mask<Coord2D>(MASK_SHAPE::ELLIPTICAL, {0., 0.}, {2., 0.2}, {});
  check_overlap(ellipse0, {MASK_SHAPE::ELLIPTICAL, {2., 0.2}}, square,
                sample_region(polygon(square), 100, 1000));

  const auto circle =
      construct_mask<Coord2D>(MASK_SHAPE::CIRCULAR, {0., 0.}, {1.5}, {});
  assert(circle.tile_overlap(square) == OVERLAP_LEVEL::PARTIAL);

  const Tile<Coord2D> rectangle(TILE_SHAPE::RECTANGLE,
                                construct_coord_2D(1., 0.5), {2., 1.}, {});
  const auto small_circle =
      construct_mask<Coord2D>(MASK_SHAPE::CIRCULAR, {0.5, -0.3}, {0.35}, {});
  assert(small_circle.tile_overlap(rectangle) == OVERLAP_LEVEL::PARTIAL);
}

int main() {
  test_catalog();
  test_regressions();
  return 0;
}
