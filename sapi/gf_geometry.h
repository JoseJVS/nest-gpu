/*
 *  gf_geometry.h
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

#ifndef GF_GEOMETRY_H
#define GF_GEOMETRY_H

#include <cassert>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "coordinates.h"
#include "enum_store.h"
#include "grid_containers.h"
#include "sapi_forward_declarations.h"

// Dimension-agnostic dispatch of the grid functor queries; the dimension- and
// shape-aware implementations are in gf2d_geometry.h (future gf3d_geometry.h),
// which grid_functors.h includes after defining the functor types, so the
// dependent calls below find them at instantiation.
namespace sapi {
template <Coordinate CoordT>
void initialize_soc(ShiftedOriginCreator<CoordT> &soc,
                    const std::vector<space_t> &side_lengths,
                    const std::vector<angle_t> &angular_offsets) {
  if constexpr (std::is_same_v<CoordT, Coord2D>) {
    initialize_shifted_origin_creator_2D(soc, side_lengths, angular_offsets);
  } else {
    throw std::runtime_error("3D implementation not available yet");
  }
}

template <Coordinate CoordT>
CoordT
create_shifted_origin(const CoordT &grid_origin,
                      const GridPosition<CoordT> &grid_position,
                      const GridPositionParity<CoordT> &grid_position_parity,
                      const ShiftedOriginCreator<CoordT> &soc) {
  if constexpr (std::is_same_v<CoordT, Coord2D>) {
    switch (soc.shape_) {
    case TILE_SHAPE::RECTANGLE:
      return create_shifted_rectangular_origin(grid_origin, grid_position, soc);

    case TILE_SHAPE::TRIANGLE:
      return create_shifted_triangular_origin(grid_origin, grid_position,
                                              grid_position_parity, soc);

    case TILE_SHAPE::HEXAGON:
      return create_shifted_hexagonal_origin(grid_origin, grid_position,
                                             grid_position_parity, soc);

    default:
      assert(false && "Invalid tile shape");
      return grid_origin;
    }
  } else {
    assert(false && "3D implementation not available yet");
    return grid_origin;
  }
}

template <Coordinate CoordT>
void initialize_ctc(CachedTileCreator<CoordT> &ctc,
                    const std::vector<space_t> &side_lengths,
                    const std::vector<angle_t> &angular_offsets) {
  if constexpr (std::is_same_v<CoordT, Coord2D>) {
    initialize_cached_tile_creator_2D(ctc, side_lengths, angular_offsets);
  } else {
    throw std::runtime_error("3D implementation not available yet");
  }
}

template <Coordinate CoordT>
bool check_dimensions(const GridPosition<CoordT> &grid_dimensions,
                      const CachedTileCreator<CoordT> &ctc) {
  if constexpr (std::is_same_v<CoordT, Coord2D>) {
    switch (ctc.shape_) {
    case TILE_SHAPE::RECTANGLE:
      return check_rectangular_dimensions(grid_dimensions, ctc);

    case TILE_SHAPE::TRIANGLE:
      return check_triangular_dimensions(grid_dimensions, ctc);

    case TILE_SHAPE::HEXAGON:
      return check_hexagonal_dimensions(grid_dimensions, ctc);

    default:
      throw std::invalid_argument("Invalid tile shape");
    }
  } else {
    throw std::runtime_error("3D implementation not available yet");
  }
}

template <Coordinate CoordT>
Tile<CoordT> create_tile(const CoordT &tile_origin,
                         const GridPositionParity<CoordT> &grid_position_parity,
                         const CachedTileCreator<CoordT> &ctc) {
  if constexpr (std::is_same_v<CoordT, Coord2D>) {
    switch (ctc.shape_) {
    case TILE_SHAPE::RECTANGLE:
      return create_rectangle(tile_origin, ctc);

    case TILE_SHAPE::TRIANGLE:
      return create_triangle(tile_origin, grid_position_parity, ctc);

    case TILE_SHAPE::HEXAGON:
      return create_hexagon(tile_origin, ctc);

    default:
      assert(false && "Invalid tile shape");
      return Tile<CoordT>{};
    }
  } else {
    assert(false && "3D implementation not available yet");
    return Tile<CoordT>{};
  }
}
} // namespace sapi

#endif
