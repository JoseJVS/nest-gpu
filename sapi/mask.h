/*
 *  mask.h
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

#ifndef MASK_H
#define MASK_H

#include <cassert>
#include <cmath>
#include <deque>
#include <utility>

#include "mask_geometry.h"
#include "sapi_forward_declarations.h"

namespace sapi {
template <Coordinate CoordT> struct Mask {
  bool has_c_radius_ = false;
  MASK_SHAPE shape_ = MASK_SHAPE::NULL_MS;
  CoordT offset_;
  CircumscribedRadius<CoordT> c_radius_;
  std::vector<CoordT> helper_vectors_;
  std::vector<space_t> helper_scalars_;

  OptDisp<CoordT> coord_in_mask(const CoordT &coord) const;

  OptDisp<CoordT> coord_in_mask(const CoordT &a, const CoordT &b) const;

  bool overlap_with_tile_surface(const Tile<CoordT> &tile) const;

  bool overlap_with_tile_surface(const CoordT &coord,
                                 const Tile<CoordT> &tile) const;

  bool tiles_within_mask_range(const Tile<CoordT> &a,
                               const Tile<CoordT> &b) const;

  OVERLAP_LEVEL tile_overlap(const Tile<CoordT> &tile) const;

  std::deque<const Tile<CoordT> *>
  get_overlapping_leaf_sub_tiles(const Tile<CoordT> &tile) const;
};

template <Coordinate CoordT>
Mask<CoordT> construct_mask(const MASK_SHAPE shape,
                            const std::vector<space_t> &mask_origin,
                            const std::vector<space_t> &mask_params,
                            const std::vector<space_t> &mask_offset) {
  if (mask_params.empty())
    throw std::invalid_argument("Invalid mask params vector");

  if (!mask_origin.empty() && !std::cmp_equal(mask_origin.size(), CoordT::D))
    throw std::invalid_argument("Invalid mask origin vector");

  if (!mask_offset.empty() && !std::cmp_equal(mask_offset.size(), CoordT::D))
    throw std::invalid_argument("Invalid mask offset vector");

  Mask<CoordT> m;
  m.shape_ = shape;

  if (!mask_offset.empty())
    m.offset_ = CoordT::copy_from_vec(mask_offset.begin());

  if (!mask_origin.empty()) {
    m.has_c_radius_ = true;
    m.c_radius_.origin_ =
        CoordT::copy_from_vec(mask_origin.begin()) + m.offset_;
  }

  initialize_mask_helpers(m, mask_params);

  return m;
}

template <Coordinate CoordT>
inline OptDisp<CoordT> Mask<CoordT>::coord_in_mask(const CoordT &coord) const {
  assert(has_c_radius_);
  return sapi::coord_in_mask(coord, *this);
}

template <Coordinate CoordT>
inline OptDisp<CoordT> Mask<CoordT>::coord_in_mask(const CoordT &a,
                                                   const CoordT &b) const {
  return sapi::coord_in_mask(a, b, *this);
}

template <Coordinate CoordT>
inline bool
Mask<CoordT>::overlap_with_tile_surface(const Tile<CoordT> &tile) const {
  assert(has_c_radius_);
  // The perimeter point closest to the mask center decides overlap exactly for
  // circles; for other shapes the circumscribed circle makes it conservative
  return c_radius_
      .coord_in_radius(tile.project_point_to_surface(c_radius_.origin_))
      .first;
}

// Two-argument (blueprint) methods: the anchor comes from the arguments, so
// only offset_ and c_radius_.radius2_ of the mask are used. Every pair the
// node test accepts lies within sqrt(radius2_) of the offset anchor, so the
// tests are exact for circular blueprints and conservative for other shapes.
template <Coordinate CoordT>
inline bool
Mask<CoordT>::overlap_with_tile_surface(const CoordT &coord,
                                        const Tile<CoordT> &tile) const {
  const CoordT shifted = coord + offset_;
  return tile.coord_in_tile(shifted) ||
         leq_test(distance2(shifted, tile.project_point_to_surface(shifted)),
                  c_radius_.radius2_);
}

template <Coordinate CoordT>
inline bool Mask<CoordT>::tiles_within_mask_range(const Tile<CoordT> &a,
                                                  const Tile<CoordT> &b) const {
  // Driver tile translated by the offset: A' = A + offset_
  // 1. Circumscribed circles of A' and B too far apart for the blueprint
  if (!a.c_radius_
           .coord_in_radius(b.c_radius_.origin_ - offset_,
                            std::sqrt(b.c_radius_.radius2_) +
                                std::sqrt(c_radius_.radius2_))
           .first)
    return false;

  // 2. Distance or overlap of A' and B within the blueprint radius (dimension-
  // aware; exact for circular blueprints, conservative for other shapes)
  return tiles_within_range(a, b, offset_, c_radius_.radius2_);
}

template <Coordinate CoordT>
OVERLAP_LEVEL Mask<CoordT>::tile_overlap(const Tile<CoordT> &tile) const {
  assert(has_c_radius_);

  if (!c_radius_.overlapping_radi(tile.c_radius_).first)
    return OVERLAP_LEVEL::NONE;

  bool fully_in_mask = true;
  vertidx_t vertices_in_mask = 0;
  auto tile_vit = tile.vertices_.cbegin();
  const auto tile_vend = tile.vertices_.cend();
  for (; tile_vit != tile_vend; ++tile_vit) {
    fully_in_mask &= coord_in_mask(*tile_vit).first;
    if (!fully_in_mask)
      break; // Stop at first failure
    ++vertices_in_mask;
  }

  if (fully_in_mask)
    return OVERLAP_LEVEL::FULL;

  if (0 < vertices_in_mask)
    return OVERLAP_LEVEL::PARTIAL;

  if (tile.coord_in_tile(c_radius_.origin_) ||
      coord_in_mask(tile.c_radius_.origin_).first)
    return OVERLAP_LEVEL::PARTIAL;

  // Continue after the vertex that failed the first loop (it always broke at
  // one, otherwise the tile would be FULL)
  for (++tile_vit; tile_vit != tile_vend; ++tile_vit)
    if (coord_in_mask(*tile_vit).first)
      return OVERLAP_LEVEL::PARTIAL; // return on first success

  if (overlap_with_tile_surface(tile))
    return OVERLAP_LEVEL::PARTIAL;

  return OVERLAP_LEVEL::NONE;
}

template <Coordinate CoordT>
std::deque<const Tile<CoordT> *>
Mask<CoordT>::get_overlapping_leaf_sub_tiles(const Tile<CoordT> &tile) const {
  assert(has_c_radius_);

  std::deque<const Tile<CoordT> *> leaf_sub_tiles;
  std::deque<const Tile<CoordT> *> temps{&tile};
  do {
    auto current = temps.front();
    temps.pop_front();

    // Get level of overlap with tile
    switch (tile_overlap(*current)) {
      // If the tile is completely within the mask we insert all of
      // its leaf sub tiles ( defaults to the tile in question if no sub tiles
      // available )
    case OVERLAP_LEVEL::FULL: {
      current->insert_leaf_sub_tiles(leaf_sub_tiles);
      break;
    }

    // If there is only a partial overlap then we check into the sub-tiles
    // if there are no sub-tiles then we add the current tile to our result
    // queue
    case OVERLAP_LEVEL::PARTIAL: {
      if (!current->sub_tiles_.empty())
        for (const auto &st : current->sub_tiles_)
          temps.emplace_back(&st);
      else
        leaf_sub_tiles.emplace_back(current);

      break;
    }

    default:
      break;
    }

  } while (!temps.empty());

  return leaf_sub_tiles;
}
} // namespace sapi

// Definitions of the 2D functions declared in mask_geometry.h. They need the
// complete Mask, so they are included after it.
#include "mask2d_geometry.h"

#endif
