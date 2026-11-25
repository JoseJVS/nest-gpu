/*
 *  mask_geometry.h
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

#ifndef MASK_GEOMETRY_H
#define MASK_GEOMETRY_H

#include <cassert>
#include <cmath>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "coordinates.h"
#include "enum_store.h"
#include "sapi_forward_declarations.h"

namespace sapi {
// Dispatches to the dimension- and shape-aware initializers, which check their
// own parameters
template <Coordinate CoordT>
void initialize_mask_helpers(Mask<CoordT> &mask,
                             const std::vector<space_t> &mask_params) {
  if constexpr (std::is_same_v<CoordT, Coord2D>) {
    switch (mask.shape_) {
    case MASK_SHAPE::CIRCULAR:
      initialize_circular_mask_helpers(mask, mask_params);
      break;

    case MASK_SHAPE::ELLIPTICAL:
      initialize_elliptical_mask_helpers(mask, mask_params);
      break;

    case MASK_SHAPE::PARALLELOGRAM:
      initialize_parallelogram_mask_helpers(mask, mask_params);
      break;

    case MASK_SHAPE::TRIANGULAR:
      initialize_triangular_mask_helpers(mask, mask_params);
      break;

    case MASK_SHAPE::ANNULAR:
      initialize_annular_mask_helpers(mask, mask_params);
      break;

    default:
      throw std::invalid_argument("Invalid mask shape");
    }
  } else {
    throw std::runtime_error("3D implementation not available yet");
  }
}

// Non-convex shapes, reached through the default case of coord_in_mask: its
// switch keeps one case per convex shape, which GCC compiles to a compare
// chain instead of a jump table in the per-pair loops
template <Coordinate CoordT>
OptDisp<CoordT> coord_in_nonconvex_mask(const CoordT &coord,
                                        const Mask<CoordT> &mask) {
  switch (mask.shape_) {
  case MASK_SHAPE::ANNULAR:
    return coord_in_annular_mask(coord, mask);

  default:
    assert(false && "Invalid mask shape");
    return {};
  }
}

template <Coordinate CoordT>
OptDisp<CoordT> coord_in_nonconvex_mask(const CoordT &a, const CoordT &b,
                                        const Mask<CoordT> &mask) {
  switch (mask.shape_) {
  case MASK_SHAPE::ANNULAR:
    return coord_in_annular_mask(a, b, mask);

  default:
    assert(false && "Invalid mask shape");
    return {};
  }
}

// The dimension- and shape-aware functions called below (initialize_*,
// coord_in_*_mask, the tile tests) are declared in the per-dimension files,
// which mask.h includes after defining Mask; the dependent calls find them at
// instantiation.
template <Coordinate CoordT>
OptDisp<CoordT> coord_in_mask(const CoordT &coord, const Mask<CoordT> &mask) {
  if constexpr (std::is_same_v<CoordT, Coord2D>) {
    switch (mask.shape_) {
    case MASK_SHAPE::CIRCULAR:
      return coord_in_circular_mask(coord, mask);

    case MASK_SHAPE::ELLIPTICAL:
      return coord_in_elliptical_mask(coord, mask);

    case MASK_SHAPE::PARALLELOGRAM:
      return coord_in_parallelogram_mask(coord, mask);

    case MASK_SHAPE::TRIANGULAR:
      return coord_in_triangular_mask(coord, mask);

    default:
      return coord_in_nonconvex_mask(coord, mask);
    }
  } else {
    assert(false && "3D implementation not available yet");
    return {};
  }
}

template <Coordinate CoordT>
OptDisp<CoordT> coord_in_mask(const CoordT &a, const CoordT &b,
                              const Mask<CoordT> &mask) {
  if constexpr (std::is_same_v<CoordT, Coord2D>) {
    switch (mask.shape_) {
    case MASK_SHAPE::CIRCULAR:
      return coord_in_circular_mask(a, b, mask);

    case MASK_SHAPE::ELLIPTICAL:
      return coord_in_elliptical_mask(a, b, mask);

    case MASK_SHAPE::PARALLELOGRAM:
      return coord_in_parallelogram_mask(a, b, mask);

    case MASK_SHAPE::TRIANGULAR:
      return coord_in_triangular_mask(a, b, mask);

    default:
      return coord_in_nonconvex_mask(a, b, mask);
    }
  } else {
    assert(false && "3D implementation not available yet");
    return {};
  }
}

// Absolute margin of the exact (GJK) tile tests: EXACT_FILTER_REL_TOL times
// the size of the given disks (mask, tiles, nodes) and of their coordinates.
// The rounding of the node tests and of the query grows with all of them, so
// the margin covers it and no pair the node test accepts is rejected; it stays
// far below real gaps.
template <typename... Disks>
inline space_t exact_filter_margin(const Disks &...disks) {
  return EXACT_FILTER_REL_TOL *
         std::sqrt(((disks.radius2_ + vector_norm2(disks.origin_)) + ...));
}

// Overlap of a source/target mask with a tile whose vertices all lie in it
template <Coordinate CoordT>
inline OVERLAP_LEVEL
overlap_with_all_vertices_in_mask(const Mask<CoordT> &mask,
                                  const Tile<CoordT> &tile) {
  if constexpr (std::is_same_v<CoordT, Coord2D>) {
    switch (mask.shape_) {
    case MASK_SHAPE::CIRCULAR:
    case MASK_SHAPE::ELLIPTICAL:
    case MASK_SHAPE::PARALLELOGRAM:
    case MASK_SHAPE::TRIANGULAR:
      return overlap_with_all_vertices_in_convex_mask(mask, tile);

    case MASK_SHAPE::ANNULAR:
      return overlap_with_all_vertices_in_annular_mask(mask, tile);

    default:
      assert(false && "Invalid mask shape");
      return OVERLAP_LEVEL::PARTIAL;
    }
  } else {
    assert(false && "3D implementation not available yet");
    return OVERLAP_LEVEL::PARTIAL;
  }
}

// Whether a source/target mask meets the tile (the last stage of
// tile_overlap)
template <Coordinate CoordT>
bool mask_meets_tile(const Mask<CoordT> &mask, const Tile<CoordT> &tile) {
  if constexpr (std::is_same_v<CoordT, Coord2D>) {
    switch (mask.shape_) {
    case MASK_SHAPE::CIRCULAR:
      return circular_mask_meets_tile(mask, tile);

    case MASK_SHAPE::ELLIPTICAL:
    case MASK_SHAPE::PARALLELOGRAM:
    case MASK_SHAPE::TRIANGULAR:
      return mask_support_meets_tile(mask, tile);

    case MASK_SHAPE::ANNULAR:
      return annular_mask_meets_tile(mask, tile);

    default:
      assert(false && "Invalid mask shape");
      return true;
    }
  } else {
    assert(false && "3D implementation not available yet");
    return true;
  }
}

// Whether the blueprint placed at the node coord meets the tile
template <Coordinate CoordT>
bool coord_tile_in_blueprint(const Mask<CoordT> &mask, const CoordT &coord,
                             const Tile<CoordT> &tile) {
  if constexpr (std::is_same_v<CoordT, Coord2D>) {
    switch (mask.shape_) {
    case MASK_SHAPE::CIRCULAR:
      return coord_tile_in_circular_blueprint(mask, coord, tile);

    case MASK_SHAPE::ELLIPTICAL:
    case MASK_SHAPE::PARALLELOGRAM:
    case MASK_SHAPE::TRIANGULAR:
      return coord_tile_in_blueprint_support(mask, coord, tile);

    case MASK_SHAPE::ANNULAR:
      return coord_tile_in_annular_blueprint(mask, coord, tile);

    default:
      assert(false && "Invalid mask shape");
      return true;
    }
  } else {
    assert(false && "3D implementation not available yet");
    return true;
  }
}

// Whether the blueprint holds a displacement b - a of the tiles a and b
template <Coordinate CoordT>
bool tile_pair_in_blueprint(const Mask<CoordT> &mask, const Tile<CoordT> &a,
                            const Tile<CoordT> &b) {
  if constexpr (std::is_same_v<CoordT, Coord2D>) {
    switch (mask.shape_) {
    case MASK_SHAPE::CIRCULAR:
      return tile_pair_in_circular_blueprint(mask, a, b);

    case MASK_SHAPE::ELLIPTICAL:
    case MASK_SHAPE::PARALLELOGRAM:
    case MASK_SHAPE::TRIANGULAR:
      return tile_pair_in_blueprint_support(mask, a, b);

    case MASK_SHAPE::ANNULAR:
      return tile_pair_in_annular_blueprint(mask, a, b);

    default:
      assert(false && "Invalid mask shape");
      return true;
    }
  } else {
    assert(false && "3D implementation not available yet");
    return true;
  }
}
} // namespace sapi

#endif
