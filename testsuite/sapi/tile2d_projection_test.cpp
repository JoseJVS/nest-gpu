/*
 *  tile2d_projection_test.cpp
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

// Tile::project_point_to_surface returns the perimeter point closest to a
// point outside the tile, for every 2D tile shape and their sub-tiles.

#include <random>

#include "coordinate_geometry.h"
#include "geometry_test_helpers.h"
#include "test_helpers.h"
#include "tile.h"

using namespace sapi;
using namespace geometry_test;

// The projection is a perimeter point at the exact distance
void check_projection(const Tile<Coord2D> &tile, const Coord2D &coord) {
  const auto v = polygon(tile);
  const P p = from(coord);
  assert(!in_convex(p, v));

  const Coord2D projected = tile.project_point_to_surface(coord);
  const double tolerance =
      64 * SP_EPSILON * std::max({1., std::fabs(p.x), std::fabs(p.y)});
  assert(std::fabs(norm(p - from(projected)) - perimeter_distance(p, v)) <=
         tolerance);
  assert(perimeter_distance(from(projected), v) <= tolerance);
}

// Points beyond each edge (nearest point on the edge) and in each vertex's
// normal cone (nearest point is the vertex), at distances from 1e-3 to 2R
void test_catalog() {
  std::deque<Tile<Coord2D>> roots;
  const auto catalog = tile_catalog(roots, 2);
  std::mt19937_64 rng(12345);
  std::uniform_real_distribution<double> unit(0, 1);

  for (const auto &nt : catalog) {
    const auto v = polygon(*nt.tile);
    const double radius = std::sqrt(nt.tile->c_radius_.radius2_);
    const auto distance = [&] {
      return std::exp(std::log(1e-3) +
                      unit(rng) * (std::log(2 * radius) - std::log(1e-3)));
    };

    for (int k = 0; k < 100; ++k) {
      const std::size_t e = rng() % v.size();
      const P q = v[e] + (v[(e + 1) % v.size()] - v[e]) * unit(rng);
      check_projection(*nt.tile,
                       to_coord(q + outward_normal(v, e) * distance()));

      const std::size_t i = rng() % v.size();
      const P n0 = outward_normal(v, (i + v.size() - 1) % v.size());
      const P n1 = outward_normal(v, i);
      // Sweep the cone in the polygon's own turning direction
      const bool ccw = 0 < orientation(v);
      const double a0 = std::atan2((ccw ? n0 : n1).y, (ccw ? n0 : n1).x);
      double a1 = std::atan2((ccw ? n1 : n0).y, (ccw ? n1 : n0).x);
      while (a1 < a0)
        a1 += 2 * M_PI;
      const double a = a0 + (a1 - a0) * unit(rng);
      check_projection(
          *nt.tile, to_coord(v[i] + P{std::cos(a), std::sin(a)} * distance()));
    }
  }
}

// Cases where the two nearest vertices don't span the nearest edge
void test_regressions() {
  const auto close = [](const Coord2D &a, const P b) {
    return std::fabs(a.x_ - b.x) < 1e-12 && std::fabs(a.y_ - b.y) < 1e-12;
  };

  const Tile<Coord2D> square(TILE_SHAPE::RECTANGLE,
                             construct_coord_2D(0.5, 0.5), {1.}, {});
  assert(close(square.project_point_to_surface(construct_coord_2D(0.3, -0.3)),
               {0.3, 0.}));

  const Tile<Coord2D> rectangle(TILE_SHAPE::RECTANGLE,
                                construct_coord_2D(1., 0.5), {2., 1.}, {});
  assert(
      close(rectangle.project_point_to_surface(construct_coord_2D(0.5, -0.3)),
            {0.5, 0.}));

  // Vertices (0.5, 0.5), (-0.5, 0.5), (-0.5, -0.5); point beyond the hypotenuse
  const Tile<Coord2D> triangle(TILE_SHAPE::TRIANGLE, construct_coord_2D(0., 0.),
                               {1.}, {});
  assert(close(triangle.project_point_to_surface(construct_coord_2D(0.3, 0.1)),
               {0.2, 0.2}));
}

int main() {
  test_catalog();
  test_regressions();
  return 0;
}
