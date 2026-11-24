/*
 *  grid_containers.h
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

#ifndef GRID_CONTAINERS_H
#define GRID_CONTAINERS_H

#include <array>
#include <cassert>
#include <cstddef>
#include <vector>

#include "sapi_config.h"
#include "sapi_forward_declarations.h"

namespace sapi {
// Relative displacement in a grid in the form of
// an integer (signed type shift_t) along each dimension of the grid
template <Coordinate CoordT> using GridShift = std::array<shift_t, CoordT::D>;

// Collection of relative displacements
template <Coordinate CoordT>
using GridShiftVector = std::vector<GridShift<CoordT>>;

// Position dependent pair GridShiftVector
// first is used of even grid positions
// second is used for odd grid positions
template <Coordinate CoordT>
using GridShiftVectorPair =
    std::pair<GridShiftVector<CoordT>, GridShiftVector<CoordT>>;

// Array of position dependent GridShiftFLs
// one GridShiftVectorPair for each dimension of the grid
template <Coordinate CoordT>
using GridDimensionalShifts =
    std::array<GridShiftVectorPair<CoordT>, CoordT::D>;

// Dimensional index decomposition of position in tile grid
template <Coordinate CoordT>
using GridPosition = std::array<tileidx_t, CoordT::D>;

// For each dimensional index
// indicates whether the position is even or odd
template <Coordinate CoordT>
using GridPositionParity = std::array<bool, CoordT::D>;

// Row-major index of a grid position in any number of dimensions:
// p0 + d0 * (p1 + d1 * (p2 + ...)), evaluated from the last dimension
template <typename T, std::size_t N>
constexpr inline T position_to_index(const std::array<T, N> &grid_position,
                                     const std::array<T, N> &grid_dimensions) {
  static_assert(0 < N);
  T index = grid_position[N - 1];
  for (std::size_t dim = N - 1; 0 < dim--;)
    index = grid_position[dim] + grid_dimensions[dim] * index;
  return index;
}

// Number of images of a tile under edge wrap: the tile shifted by each of the
// IMAGE_SHIFTS along each dimension
template <Coordinate CoordT> constexpr inline std::size_t max_tile_images() {
  std::size_t images = 1;
  for (dim_t dim = 0; dim < CoordT::D; ++dim)
    images *= IMAGE_SHIFTS.size();
  return images;
}

constexpr inline std::pair<tileidx_t, bool>
edge_wrapped_shift(const tileidx_t origin, const tileidx_t shift,
                   const tileidx_t upper_edge) {
  assert(0 <= origin && origin < upper_edge);

  const auto shifted_origin = origin + shift;
  if (shift < 0) {
    if (shifted_origin < 0) {
      const auto so_mod_u = shifted_origin % upper_edge;
      return {upper_edge * (so_mod_u != 0) + so_mod_u, true};
    } else
      return {shifted_origin, false};
  } else {
    if (upper_edge <= shifted_origin)
      return {shifted_origin % upper_edge, true};
    else
      return {shifted_origin, false};
  }
}
} // namespace sapi

#endif
