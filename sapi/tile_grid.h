/*
 *  tile_grid.h
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

#ifndef TILE_GRID_H
#define TILE_GRID_H

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "grid_containers.h"
#include "sapi_forward_declarations.h"

namespace sapi {
template <Coordinate CoordT> struct TilePosition {
  GridPosition<CoordT> position_;
  mutable Tile<CoordT> tile_;
  mutable OmpLock<CoordT> tile_lock_;
  mutable std::vector<Tile<CoordT>> tile_images_;
  mutable std::vector<OmpLock<CoordT>> image_locks_;
  std::vector<CoordT> image_displacements_;
  // Sorted and free of duplicates (see sort_neighborhood), so lookups use a
  // binary search and equal neighborhoods compare equal
  std::vector<tileidx_t> direct_tile_neighborhood_;
  std::vector<tileidx_t> wrapped_tile_neighborhood_;

  TilePosition() noexcept = default;
  TilePosition(const TilePosition &) = delete;
  TilePosition(TilePosition &&) noexcept = default;
  ~TilePosition() noexcept = default;

  TilePosition &operator=(const TilePosition &) = delete;
  TilePosition &operator=(TilePosition &&) = delete;

  const std::vector<tileidx_t> &get_tile_neighborhood(const bool) const;

  bool in_neighborhood(const tileidx_t, const bool) const;

  std::string to_string() const;

  bool operator==(const TilePosition &) const;
};

template <Coordinate CoordT>
inline const std::vector<tileidx_t> &
TilePosition<CoordT>::get_tile_neighborhood(const bool edge_wrap) const {
  return edge_wrap ? wrapped_tile_neighborhood_ : direct_tile_neighborhood_;
}

template <Coordinate CoordT>
inline bool TilePosition<CoordT>::in_neighborhood(const tileidx_t index,
                                                  const bool edge_wrap) const {
  return std::ranges::binary_search(get_tile_neighborhood(edge_wrap), index);
}

template <Coordinate CoordT>
std::string TilePosition<CoordT>::to_string() const {
  std::string res = tile_.c_radius_.origin_.to_string() + ": [ ";
  auto num_neighbors = wrapped_tile_neighborhood_.size();
  for (const auto &neighbor_index : wrapped_tile_neighborhood_)
    res += std::to_string(neighbor_index) + (num_neighbors-- > 1 ? ", " : " ]");

  return res;
}

template <Coordinate CoordT>
inline bool TilePosition<CoordT>::operator==(const TilePosition &tp) const {
  return position_ == tp.position_ &&
         direct_tile_neighborhood_ == tp.direct_tile_neighborhood_ &&
         wrapped_tile_neighborhood_ == tp.wrapped_tile_neighborhood_ &&
         image_displacements_ == tp.image_displacements_ && tile_ == tp.tile_ &&
         tile_images_ == tp.tile_images_;
}

template <Coordinate CoordT> struct TileGrid {
  bool has_split_ = false;
  split_t splits_ = 0;
  tileidx_t num_tiles_ = 0;
  GridPosition<CoordT> dimensions_;
  BoundingBox<CoordT> bounding_box_;
  std::vector<TilePosition<CoordT>> positions_;

  TileGrid() noexcept = default;
  TileGrid(const TileGrid &) = delete;
  TileGrid(TileGrid &&) noexcept = default;
  ~TileGrid() noexcept = default;

  TileGrid &operator=(const TileGrid &) = delete;
  TileGrid &operator=(TileGrid &&) noexcept;

  void prepare(const GridPosition<CoordT> &);

  std::string to_string() const;

  bool operator==(const TileGrid &) const;
};

template <Coordinate CoordT>
TileGrid<CoordT> &TileGrid<CoordT>::operator=(TileGrid &&tg) noexcept {
  if (this == &tg)
    return *this;

  has_split_ = tg.has_split_;
  tg.has_split_ = false;

  splits_ = tg.splits_;
  tg.splits_ = 0;

  num_tiles_ = tg.num_tiles_;
  tg.num_tiles_ = 0;

  auto dim_it = dimensions_.begin();
  for (auto &d : tg.dimensions_) {
    (*dim_it++) = d;
    d = 0;
  }

  bounding_box_ = std::move(tg.bounding_box_);

  positions_.swap(tg.positions_);
  tg.positions_.clear();

  return *this;
}

template <Coordinate CoordT>
void TileGrid<CoordT>::prepare(const GridPosition<CoordT> &dimensions) {
  assert(positions_.empty());
  num_tiles_ = 1;
  dimensions_ = dimensions;

  for (const auto &d : dimensions_)
    num_tiles_ *= d;
  if (num_tiles_ < 1)
    throw std::invalid_argument("Invalid grid dimensions");

  splits_ = 0;
  has_split_ = false;
  positions_.resize(static_cast<std::size_t>(num_tiles_));
}

template <Coordinate CoordT> std::string TileGrid<CoordT>::to_string() const {
  std::string res = "";
  tileidx_t idx = 0;
  for (const auto &pos : positions_)
    res += std::to_string(idx++) + ": " + pos.to_string() + "\n";

  return res;
}

template <Coordinate CoordT>
inline bool TileGrid<CoordT>::operator==(const TileGrid &tg) const {
  if (dimensions_ != tg.dimensions_)
    return false;

  // if dimensions are equal then
  // total number of tiles and position
  // vector sizes should be equal
  assert(num_tiles_ == tg.num_tiles_ &&
         positions_.size() == tg.positions_.size());

  return positions_ == tg.positions_;
}
} // namespace sapi

#endif
