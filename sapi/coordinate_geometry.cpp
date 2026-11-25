/*
 *  coordinate_geometry.cpp
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

#include <array>
#include <cassert>
#include <limits>
#include <span>

#include "coordinate_geometry.h"

namespace sapi {
Coord2D
project_point_to_polygon_perimeter(const Coord2D &coord,
                                   const std::span<const Coord2D> &vertices) {
  assert(2 < vertices.size());
  for (const auto &vertex : vertices)
    if (almost_zero(distance2(coord, vertex)))
      return vertex;

  Coord2D closest = vertices[0];
  space_t closest_d2 = std::numeric_limits<space_t>::max();
  space_t first_d2 = distance2(coord, vertices[0]);
  for (std::size_t first = 0; first < vertices.size(); ++first) {
    const std::size_t second = (first + 1) % vertices.size();
    const space_t second_d2 = distance2(coord, vertices[second]);
    const bool second_is_base = std::isless(second_d2, first_d2) ||
                                (second_d2 == first_d2 && second < first);
    const Coord2D projected = projection_coord<Coord2D, true>(
        coord, vertices[second_is_base ? second : first],
        vertices[second_is_base ? first : second]);
    const space_t d2 = distance2(coord, projected);
    if (std::isless(d2, closest_d2)) {
      closest_d2 = d2;
      closest = projected;
    }
    first_d2 = second_d2;
  }

  return closest;
}

Coord3D closest_point_on_tetrahedron_to_origin(Simplex<Coord3D> &simplex) {
  // The simplex is only replaced at the end, after its last use through p
  const auto &p = simplex.points_;
  // Faces i, j, k and the vertex l opposite to each
  constexpr std::array<std::array<std::size_t, 4>, 4> faces{
      {{0, 1, 2, 3}, {0, 1, 3, 2}, {0, 2, 3, 1}, {1, 2, 3, 0}}};

  Simplex<Coord3D> feature;
  Coord3D closest;
  space_t closest2 = std::numeric_limits<space_t>::infinity();
  for (const auto &[i, j, k, l] : faces) {
    // Only faces with the origin strictly beyond them (opposite to l) hold
    // the closest point; a flat tetrahedron has none
    const Coord3D normal = vector_cross(p[j] - p[i], p[k] - p[i]);
    if (!(0 < vector_dot(p[i], normal) * vector_dot(p[l] - p[i], normal)))
      continue;

    Simplex<Coord3D> face;
    face.assign(p[i], p[j], p[k]);
    const Coord3D point = closest_point_on_triangle_to_origin(face);
    const space_t distance2 = vector_norm2(point);
    if (distance2 < closest2) {
      closest2 = distance2;
      closest = point;
      feature = face;
    }
  }

  // Inside (or flat, or not comparable): keep the tetrahedron
  if (feature.size_ == 0)
    return Coord3D();

  simplex = feature;
  return closest;
}
} // namespace sapi
