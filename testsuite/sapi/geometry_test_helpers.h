/*
 *  geometry_test_helpers.h
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

#ifndef GEOMETRY_TEST_HELPERS_H
#define GEOMETRY_TEST_HELPERS_H

#include <algorithm>
#include <cmath>
#include <deque>
#include <string>
#include <vector>

#include "coordinate_geometry.h"
#include "tile.h"

// Reference geometry for the geometry tests, computed independently of SAPI
namespace geometry_test {
using sapi::Coord2D;
using sapi::Tile;

struct P {
  double x, y;
};

inline P operator-(const P a, const P b) { return {a.x - b.x, a.y - b.y}; }
inline P operator+(const P a, const P b) { return {a.x + b.x, a.y + b.y}; }
inline P operator*(const P a, const double s) { return {a.x * s, a.y * s}; }
inline double dot(const P a, const P b) { return a.x * b.x + a.y * b.y; }
inline double cross(const P a, const P b) { return a.x * b.y - a.y * b.x; }
inline double norm(const P a) { return std::sqrt(dot(a, a)); }
inline P from(const Coord2D &c) { return {c.x_, c.y_}; }
inline Coord2D to_coord(const P p) {
  return sapi::construct_coord_2D(p.x, p.y);
}

inline std::vector<P> polygon(const Tile<Coord2D> &t) {
  std::vector<P> v;
  for (const auto &c : t.vertices_)
    v.push_back(from(c));
  return v;
}

inline P segment_nearest(const P p, const P a, const P b) {
  const P ab = b - a;
  const double l2 = dot(ab, ab);
  const double u = l2 > 0 ? std::clamp(dot(p - a, ab) / l2, 0., 1.) : 0.;
  return a + ab * u;
}

inline double orientation(const std::vector<P> &v) {
  double area = 0;
  for (std::size_t i = 0; i < v.size(); ++i)
    area += cross(v[i], v[(i + 1) % v.size()]);
  return area > 0 ? 1 : -1;
}

inline bool in_convex(const P p, const std::vector<P> &v) {
  const double o = orientation(v);
  for (std::size_t i = 0; i < v.size(); ++i)
    if (o * cross(v[(i + 1) % v.size()] - v[i], p - v[i]) < 0)
      return false;
  return true;
}

// Distance from p to the polygon perimeter
inline double perimeter_distance(const P p, const std::vector<P> &v) {
  double best = 1e300;
  for (std::size_t i = 0; i < v.size(); ++i)
    best = std::min(best,
                    norm(p - segment_nearest(p, v[i], v[(i + 1) % v.size()])));
  return best;
}

// Distance from p to the polygon region (0 inside)
inline double region_distance(const P p, const std::vector<P> &v) {
  return in_convex(p, v) ? 0. : perimeter_distance(p, v);
}

inline P outward_normal(const std::vector<P> &v, const std::size_t edge) {
  const P e = v[(edge + 1) % v.size()] - v[edge];
  return P{e.y, -e.x} * (orientation(v) / norm(e));
}

// Interior grid points and perimeter points (uniform by length) of a polygon
inline std::vector<Coord2D> sample_region(const std::vector<P> &v,
                                          const int grid, const int perimeter) {
  std::vector<Coord2D> s;
  double x0 = 1e300, x1 = -1e300, y0 = 1e300, y1 = -1e300;
  for (const auto &p : v) {
    x0 = std::min(x0, p.x), x1 = std::max(x1, p.x);
    y0 = std::min(y0, p.y), y1 = std::max(y1, p.y);
  }
  for (int i = 0; i < grid; ++i)
    for (int j = 0; j < grid; ++j) {
      const P p{x0 + (x1 - x0) * (i + 0.5) / grid,
                y0 + (y1 - y0) * (j + 0.5) / grid};
      if (in_convex(p, v))
        s.push_back(to_coord(p));
    }
  double length = 0;
  for (std::size_t i = 0; i < v.size(); ++i)
    length += norm(v[(i + 1) % v.size()] - v[i]);
  for (int k = 0; k < perimeter; ++k) {
    double at = length * (k + 0.5) / perimeter;
    for (std::size_t i = 0; i < v.size(); ++i) {
      const P a = v[i], b = v[(i + 1) % v.size()];
      const double l = norm(b - a);
      if (at <= l) {
        s.push_back(to_coord(a + (b - a) * (at / l)));
        break;
      }
      at -= l;
    }
  }
  for (const auto &p : v)
    s.push_back(to_coord(p));
  return s;
}

struct NamedTile {
  std::string name;
  const Tile<Coord2D> *tile;
};

// Unit square, 2x1 rectangle (plain and rotated 30 degrees), both triangle
// orientations and a hexagon, with their sub-tiles down to max_level
inline std::vector<NamedTile> tile_catalog(std::deque<Tile<Coord2D>> &roots,
                                           const int max_level) {
  using sapi::TILE_SHAPE;
  struct Root {
    std::string name;
    TILE_SHAPE shape;
    Coord2D origin;
    std::vector<sapi::space_t> sides;
    std::vector<sapi::angle_t> angles;
  };
  const std::vector<Root> spec{{"square",
                                TILE_SHAPE::RECTANGLE,
                                sapi::construct_coord_2D(0.3, -0.2),
                                {1.},
                                {}},
                               {"rect21",
                                TILE_SHAPE::RECTANGLE,
                                sapi::construct_coord_2D(1., 0.5),
                                {2., 1.},
                                {}},
                               {"rect21_rot30",
                                TILE_SHAPE::RECTANGLE,
                                sapi::construct_coord_2D(-0.4, 0.7),
                                {2., 1.},
                                {30}},
                               {"triangle_0",
                                TILE_SHAPE::TRIANGLE,
                                sapi::construct_coord_2D(0.5, 0.5),
                                {1.},
                                {0}},
                               {"triangle_180",
                                TILE_SHAPE::TRIANGLE,
                                sapi::construct_coord_2D(1., 0.5),
                                {1.},
                                {180}},
                               {"hexagon",
                                TILE_SHAPE::HEXAGON,
                                sapi::construct_coord_2D(0.2, 0.1),
                                {1.},
                                {30}}};

  std::vector<NamedTile> catalog;
  for (const auto &r : spec) {
    roots.emplace_back(r.shape, r.origin, r.sides, r.angles);
    roots.back().split(2, false);
    std::vector<NamedTile> level{{r.name, &roots.back()}};
    for (int l = 0; l <= max_level; ++l) {
      catalog.insert(catalog.end(), level.begin(), level.end());
      std::vector<NamedTile> next;
      for (const auto &nt : level) {
        std::size_t i = 0;
        for (const auto &st : nt.tile->sub_tiles_)
          next.push_back({nt.name + "." + std::to_string(i++), &st});
      }
      level = next;
    }
  }
  return catalog;
}
} // namespace geometry_test

#endif
