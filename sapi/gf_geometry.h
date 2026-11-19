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

#include "coordinates.h"
#include "enum_store.h"
#include "grid_containers.h"
#include "sapi_forward_declarations.h"
#include "tile2d_geometry.h"

namespace sapi {
void initialize_soc_2D(ShiftedOriginCreator<Coord2D> &soc,
                       const std::vector<space_t> &side_lengths,
                       const angle_t angular_offset);

void initialize_hexagonal_soc(ShiftedOriginCreator<Coord2D> &soc,
                              const std::vector<space_t> &side_lengths,
                              const angle_t angular_offset);

template <Coordinate CoordT>
void initialize_soc(ShiftedOriginCreator<CoordT> &soc,
                    const std::vector<space_t> &side_lengths,
                    const std::vector<angle_t> &angular_offsets) {
  if constexpr (std::is_same_v<CoordT, Coord2D>) {
    const angle_t angular_offset =
        validate_tile_parameters_2D(soc.shape_, side_lengths, angular_offsets);

    switch (soc.shape_) {
    case TILE_SHAPE::RECTANGLE:
    case TILE_SHAPE::TRIANGLE:
      initialize_soc_2D(soc, side_lengths, angular_offset);
      break;

    case TILE_SHAPE::HEXAGON:
      initialize_hexagonal_soc(soc, side_lengths, angular_offset);
      break;

    default:
      throw std::invalid_argument("Invalid tile shape");
    }
  } else {
    throw std::runtime_error("3D implementation not available yet");
  }
}

Coord2D
create_shifted_rectangular_origin(const Coord2D &grid_origin,
                                  const GridPosition<Coord2D> &grid_position,
                                  const ShiftedOriginCreator<Coord2D> &soc);

Coord2D create_shifted_triangular_origin(
    const Coord2D &grid_origin, const GridPosition<Coord2D> &grid_position,
    const GridPositionParity<Coord2D> &grid_position_parity,
    const ShiftedOriginCreator<Coord2D> &soc);

Coord2D create_shifted_hexagonal_origin(
    const Coord2D &grid_origin, const GridPosition<Coord2D> &grid_position,
    const GridPositionParity<Coord2D> &grid_position_parity,
    const ShiftedOriginCreator<Coord2D> &soc);

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

void initialize_rectangular_ctc(CachedTileCreator<Coord2D> &ctc,
                                const std::vector<space_t> &side_lengths,
                                const angle_t angular_offset);

void initialize_triangular_ctc(CachedTileCreator<Coord2D> &ctc,
                               const std::vector<space_t> &side_lengths,
                               const angle_t angular_offset);

void initialize_hexagonal_ctc(CachedTileCreator<Coord2D> &ctc,
                              const std::vector<space_t> &side_lengths,
                              const angle_t angular_offset);

template <Coordinate CoordT>
void initialize_ctc(CachedTileCreator<CoordT> &ctc,
                    const std::vector<space_t> &side_lengths,
                    const std::vector<angle_t> &angular_offsets) {
  if constexpr (std::is_same_v<CoordT, Coord2D>) {
    const angle_t angular_offset =
        validate_tile_parameters_2D(ctc.shape_, side_lengths, angular_offsets);

    switch (ctc.shape_) {
    case TILE_SHAPE::RECTANGLE:
      initialize_rectangular_ctc(ctc, side_lengths, angular_offset);
      break;

    case TILE_SHAPE::TRIANGLE:
      initialize_triangular_ctc(ctc, side_lengths, angular_offset);
      break;

    case TILE_SHAPE::HEXAGON:
      initialize_hexagonal_ctc(ctc, side_lengths, angular_offset);
      break;

    default:
      throw std::invalid_argument("Invalid tile shape");
    }
  } else {
    throw std::runtime_error("3D implementation not available yet");
  }
}

bool check_rectangular_dimensions(const GridPosition<Coord2D> &grid_dimensions,
                                  const CachedTileCreator<Coord2D> &ctc);

bool check_triangular_dimensions(const GridPosition<Coord2D> &grid_dimensions,
                                 const CachedTileCreator<Coord2D> &ctc);

bool check_hexagonal_dimensions(const GridPosition<Coord2D> &grid_dimensions,
                                const CachedTileCreator<Coord2D> &ctc);

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

Tile<Coord2D> create_rectangle(const Coord2D &tile_origin,
                               const CachedTileCreator<Coord2D> &ctc);

Tile<Coord2D>
create_triangle(const Coord2D &tile_origin,
                const GridPositionParity<Coord2D> &grid_position_parity,
                const CachedTileCreator<Coord2D> &ctc);

Tile<Coord2D> create_hexagon(const Coord2D &tile_origin,
                             const CachedTileCreator<Coord2D> &ctc);

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
