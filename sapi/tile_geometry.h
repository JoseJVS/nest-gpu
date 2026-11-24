/*
 *  tile_geometry.h
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

#ifndef TILE_GEOMETRY_H
#define TILE_GEOMETRY_H

#include <cassert>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "enum_store.h"
#include "sapi_forward_declarations.h"
#include "tile2d_geometry.h"

namespace sapi {
// The tile functions below run inside parallel regions (grid generation,
// splitting, node generation, insertion, slicing, mask overlap). Shapes come
// from checked registry keys and no 3D manager can be built, so their default
// and 3D branches assert instead of throwing.
template <Coordinate CoordT>
void initialize_tile_vertices(std::vector<CoordT> &vertices,
                              CircumscribedRadius<CoordT> &c_radius,
                              const std::vector<space_t> &side_lengths,
                              const std::vector<angle_t> &angular_offsets,
                              const TILE_SHAPE shape) {
  if constexpr (std::is_same_v<CoordT, Coord2D>) {
    initialize_tile_vertices_2D(vertices, c_radius, side_lengths,
                                angular_offsets, shape);
  } else {
    assert(false && "3D implementation not available yet");
  }
}

template <Coordinate CoordT>
void initialize_tile_helpers(std::vector<CoordT> &helper_vectors,
                             std::vector<space_t> &helper_scalars,
                             const std::vector<CoordT> &vertices,
                             const TILE_SHAPE shape) {
  if constexpr (std::is_same_v<CoordT, Coord2D>) {
    switch (shape) {
    case TILE_SHAPE::RECTANGLE: {
      initialize_helpers_2D(helper_vectors, helper_scalars, vertices);

      break;
    }

    case TILE_SHAPE::TRIANGLE: {
      initialize_helpers_2D(helper_vectors, helper_scalars, vertices);

      break;
    }

    case TILE_SHAPE::HEXAGON: {
      initialize_hexagon_helpers(helper_vectors, helper_scalars, vertices);

      break;
    }

    default:
      assert(false && "Invalid tile shape");
    }
  } else {
    assert(false && "3D implementation not available yet");
  }
}

template <Coordinate CoordT>
void split_tile(std::vector<Tile<CoordT>> &sub_tiles,
                std::vector<const Tile<CoordT> *> &leaf_tiles,
                const std::vector<CoordT> &vertices,
                const CircumscribedRadius<CoordT> &c_radius,
                const split_t splits, const bool generate_total_leaves_vector,
                const TILE_SHAPE shape) {
  if constexpr (std::is_same_v<CoordT, Coord2D>) {
    switch (shape) {
    case TILE_SHAPE::RECTANGLE: {
      split_rectangle(sub_tiles, leaf_tiles, vertices, c_radius, 0, splits,
                      generate_total_leaves_vector);

      break;
    }

    case TILE_SHAPE::TRIANGLE: {
      split_triangle(sub_tiles, leaf_tiles, vertices, 0, splits,
                     generate_total_leaves_vector);

      break;
    }

    case TILE_SHAPE::HEXAGON: {
      split_hexagon(sub_tiles, leaf_tiles, vertices, c_radius, 0, splits,
                    generate_total_leaves_vector);

      break;
    }

    default:
      assert(false && "Invalid tile shape");
    }
  } else {
    assert(false && "3D implementation not available yet");
  }
}

template <Coordinate CoordT>
bool coord_in_tile(const CoordT &coord,
                   const CircumscribedRadius<CoordT> &c_radius,
                   const std::vector<CoordT> &vertices,
                   const std::vector<CoordT> &helper_vectors,
                   const std::vector<space_t> &helper_scalars,
                   const TILE_SHAPE shape) {
  if constexpr (std::is_same_v<CoordT, Coord2D>) {
    switch (shape) {
    case TILE_SHAPE::RECTANGLE:
      return coord_in_rectangle(coord, c_radius, vertices, helper_vectors,
                                helper_scalars);

    case TILE_SHAPE::TRIANGLE:
      return coord_in_triangle(coord, c_radius, vertices, helper_vectors,
                               helper_scalars);

    case TILE_SHAPE::HEXAGON:
      return coord_in_hexagon(coord, c_radius, vertices, helper_vectors,
                              helper_scalars);

    default:
      assert(false && "Invalid tile shape");
      return false;
    }
  } else {
    assert(false && "3D implementation not available yet");
    return false;
  }
}

template <Coordinate CoordT>
std::vector<std::pair<nodeidx_t, CoordT>> generate_coords_in_tile(
    const nodeidx_t first_index, const nodeidx_t coord_count, AnyRNG &rng,
    const std::vector<CoordT> &vertices,
    const std::vector<CoordT> &helper_vectors,
    const CircumscribedRadius<CoordT> &c_radius, const TILE_SHAPE shape) {
  if constexpr (std::is_same_v<CoordT, Coord2D>) {
    switch (shape) {
    case TILE_SHAPE::RECTANGLE:
      return generate_coords_in_rectangle(first_index, coord_count, rng,
                                          vertices, helper_vectors);

    case TILE_SHAPE::TRIANGLE:
      return generate_coords_in_triangle(first_index, coord_count, rng,
                                         vertices);

    case TILE_SHAPE::HEXAGON:
      return generate_coords_in_hexagon(first_index, coord_count, rng, vertices,
                                        c_radius);

    default:
      assert(false && "Invalid tile shape");
      return {};
    }
  } else {
    assert(false && "3D implementation not available yet");
    return {};
  }
}

template <Coordinate CoordT>
CoordT project_point_to_surface(const CoordT &coord,
                                const std::vector<CoordT> &vertices,
                                const CircumscribedRadius<CoordT> &c_radius,
                                const TILE_SHAPE shape) {
  if constexpr (std::is_same_v<CoordT, Coord2D>) {
    switch (shape) {
    case TILE_SHAPE::RECTANGLE:
      return project_point_to_polygon_perimeter(coord, vertices);

    case TILE_SHAPE::HEXAGON:
      return project_point_to_polygon_perimeter(coord, vertices);

    case TILE_SHAPE::TRIANGLE:
      return project_point_to_triangle_perimeter(coord, vertices, c_radius);

    default:
      assert(false && "Invalid tile shape");
      return coord;
    }
  } else {
    assert(false && "3D implementation not available yet");
    return coord;
  }
}

// Whether some displacement from the driver tile translated by the offset,
// A' = A + offset, to the pool tile B is within sqrt(radius2): their distance
// is within the radius, or they overlap. Called inside parallel regions: the
// unimplemented 3D branch asserts instead of throwing
template <Coordinate CoordT>
bool tiles_within_range(const Tile<CoordT> &a, const Tile<CoordT> &b,
                        const CoordT &offset, const space_t radius2) {
  if constexpr (std::is_same_v<CoordT, Coord2D>) {
    return tiles_within_range_2D(a, b, offset, radius2);
  } else {
    assert(false && "3D implementation not available yet");
    return true;
  }
}

// Number of leaves of a tile of the given shape after splits splits
template <Coordinate CoordT>
tileidx_t compute_leaves_count(const split_t splits, const TILE_SHAPE shape) {
  if constexpr (std::is_same_v<CoordT, Coord2D>) {
    return compute_leaves_count_2D(splits, shape);
  } else {
    assert(false && "3D implementation not available yet");
    return 1;
  }
}

// Sub-tile fan-out of each of the splits splits
template <Coordinate CoordT>
std::vector<split_t> get_possible_sub_tile_branches(const split_t splits,
                                                    const TILE_SHAPE shape) {
  if constexpr (std::is_same_v<CoordT, Coord2D>) {
    return get_possible_sub_tile_branches_2D(splits, shape);
  } else {
    assert(false && "3D implementation not available yet");
    return {};
  }
}

// Largest number of splits whose leaf count fits tileidx_t
template <Coordinate CoordT> split_t max_splits(const TILE_SHAPE shape) {
  const auto branches = get_possible_sub_tile_branches<CoordT>(
      std::numeric_limits<split_t>::max(), shape);

  // Leaf count after each split, in a wider type than tileidx_t
  int64_t leaves = 1;
  split_t splits = 0;
  for (const auto fan_out : branches) {
    leaves *= fan_out;
    if (std::numeric_limits<tileidx_t>::max() < leaves)
      break;
    ++splits;
  }

  return splits;
}

// Fewest splits so that leaves hold about expected_nodes_per_leaf nodes
template <Coordinate CoordT>
split_t compute_minimal_splits(const double expected_total_nodes,
                               const double total_tiles,
                               const double expected_nodes_per_leaf,
                               const TILE_SHAPE shape) {
  if (expected_total_nodes < 1 || total_tiles < 1 ||
      expected_nodes_per_leaf < 1)
    throw std::invalid_argument("Invalid split parameters");

  if constexpr (std::is_same_v<CoordT, Coord2D>) {
    return compute_minimal_splits_2D(expected_total_nodes, total_tiles,
                                     expected_nodes_per_leaf, shape);
  } else {
    throw std::runtime_error("3D implementation not available yet");
  }
}
} // namespace sapi

#endif
