/*
 *  tile2d_geometry.h
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

#ifndef TILE2D_GEOMETRY_H
#define TILE2D_GEOMETRY_H

#include <stdexcept>
#include <vector>

#include "coordinates.h"
#include "enum_store.h"
#include "numerics.h"
#include "sapi_forward_declarations.h"

namespace sapi {
// Checks the 2D parameters shared by tiles and grid functors: at most one
// angular offset, then the number and the values of the side lengths of the
// shape (the same conditions the tile vertices are built under, see
// initialize_vertices_2D and initialize_hexagon_vertices). Returns the angular
// offset, 0 when none is given.
inline angle_t
validate_tile_parameters_2D(const TILE_SHAPE shape,
                            const std::vector<space_t> &side_lengths,
                            const std::vector<angle_t> &angular_offsets) {
  if (1 < angular_offsets.size())
    throw std::invalid_argument("Invalid offsets for tile 2D");
  const auto lengths = side_lengths.size();

  switch (shape) {
  case TILE_SHAPE::RECTANGLE:
    if (lengths < 1 || 2 < lengths)
      throw std::invalid_argument("Invalid rectangle length vector");
    break;

  case TILE_SHAPE::TRIANGLE:
    if (lengths < 1 || 2 < lengths)
      throw std::invalid_argument("Invalid triangle length vector");
    break;

  case TILE_SHAPE::HEXAGON:
    if (lengths != 1)
      throw std::invalid_argument("Invalid hexagon length vector");
    break;

  default:
    throw std::invalid_argument("Invalid tile shape");
  }

  for (const auto side : side_lengths)
    if (almost_zero(side * side) || std::signbit(side))
      throw std::invalid_argument(shape == TILE_SHAPE::HEXAGON
                                      ? "Invalid hexagon side length"
                                      : "Invalid 2D side length");

  // Circumscribed radius of a rectangle or triangle: half its diagonal
  if (shape != TILE_SHAPE::HEXAGON) {
    const auto half_width = side_lengths[0] / 2;
    const auto half_height = 1 < lengths ? side_lengths[1] / 2 : half_width;
    if (almost_zero(half_width * half_width + half_height * half_height))
      throw std::invalid_argument("Invalid 2D radius");
  }

  return angular_offsets.empty() ? angle_t(0) : angular_offsets[0];
}

void initialize_vertices_2D(std::vector<Coord2D> &vertices,
                            CircumscribedRadius<Coord2D> &c_radius,
                            const std::vector<space_t> &side_lengths,
                            const angle_t angular_offset,
                            const bool triangular_vertices);

void initialize_hexagon_vertices(std::vector<Coord2D> &vertices,
                                 CircumscribedRadius<Coord2D> &c_radius,
                                 const space_t side_length,
                                 const angle_t angular_offset);

void initialize_helpers_2D(std::vector<Coord2D> &helper_vectors,
                           std::vector<space_t> &helper_scalars,
                           const std::vector<Coord2D> &vertices);

void initialize_hexagon_helpers(std::vector<Coord2D> &helper_vectors,
                                std::vector<space_t> &helper_scalars,
                                const std::vector<Coord2D> &vertices);

void split_rectangle(std::vector<Tile<Coord2D>> &sub_tiles,
                     std::vector<const Tile<Coord2D> *> &leaf_tiles,
                     const std::vector<Coord2D> &vertices,
                     const CircumscribedRadius<Coord2D> &c_radius,
                     const tileidx_t prefix, const split_t splits,
                     const bool generate_total_leaves_vector);

void split_triangle(std::vector<Tile<Coord2D>> &sub_tiles,
                    std::vector<const Tile<Coord2D> *> &leaf_tiles,
                    const std::vector<Coord2D> &vertices,
                    const tileidx_t prefix, const split_t splits,
                    const bool generate_total_leaves_vector);

void split_hexagon(std::vector<Tile<Coord2D>> &sub_tiles,
                   std::vector<const Tile<Coord2D> *> &leaf_tiles,
                   const std::vector<Coord2D> &vertices,
                   const CircumscribedRadius<Coord2D> &c_radius,
                   const tileidx_t prefix, const split_t splits,
                   const bool generate_total_leaves_vector);

bool coord_in_rectangle(const Coord2D &coord,
                        const CircumscribedRadius<Coord2D> &c_radius,
                        const std::vector<Coord2D> &vertices,
                        const std::vector<Coord2D> &helper_vectors,
                        const std::vector<space_t> &helper_scalars);

bool coord_in_triangle(const Coord2D &coord,
                       const CircumscribedRadius<Coord2D> &c_radius,
                       const std::vector<Coord2D> &vertices,
                       const std::vector<Coord2D> &helper_vectors,
                       const std::vector<space_t> &helper_scalars);

bool coord_in_hexagon(const Coord2D &coord,
                      const CircumscribedRadius<Coord2D> &c_radius,
                      const std::vector<Coord2D> &vertices,
                      const std::vector<Coord2D> &helper_vectors,
                      const std::vector<space_t> &helper_scalars);

std::vector<std::pair<nodeidx_t, Coord2D>>
generate_coords_in_rectangle(const nodeidx_t first_index,
                             const nodeidx_t coord_count, AnyRNG &rng,
                             const std::vector<Coord2D> &vertices,
                             const std::vector<Coord2D> &helper_vectors);

std::vector<std::pair<nodeidx_t, Coord2D>>
generate_coords_in_triangle(const nodeidx_t first_index,
                            const nodeidx_t coord_count, AnyRNG &rng,
                            const std::vector<Coord2D> &vertices);

std::vector<std::pair<nodeidx_t, Coord2D>>
generate_coords_in_hexagon(const nodeidx_t first_index,
                           const nodeidx_t coord_count, AnyRNG &rng,
                           const std::vector<Coord2D> &vertices,
                           const CircumscribedRadius<Coord2D> &c_radius);

Coord2D
project_point_to_polygon_perimeter(const Coord2D &coord,
                                   const std::vector<Coord2D> &vertices);

Coord2D project_point_to_triangle_perimeter(
    const Coord2D &coord, const std::vector<Coord2D> &vertices,
    const CircumscribedRadius<Coord2D> &c_radius);

bool tiles_within_range_2D(const Tile<Coord2D> &a, const Tile<Coord2D> &b,
                           const Coord2D &offset, const space_t radius2);

space_t min_vertex_edge_distance2_2D(const std::vector<Coord2D> &from,
                                     const Coord2D &shift,
                                     const std::vector<Coord2D> &to);

bool edges_cross_2D(const std::vector<Coord2D> &a, const Coord2D &shift,
                    const std::vector<Coord2D> &b);
} // namespace sapi

#endif
