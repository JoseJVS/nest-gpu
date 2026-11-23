/*
 *  grid_neighborhood.h
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

#ifndef GRID_NEIGHBORHOOD_H
#define GRID_NEIGHBORHOOD_H

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

#include "sapi_forward_declarations.h"
#include "tile_grid.h"
#include "vp_interface.h"

namespace sapi {

struct GridNeighborhood {
  using TileSet = std::vector<std::set<tileidx_t>>::const_iterator;
  using RankSet =
      std::vector<std::pair<std::set<vp_t>, std::set<vp_t>>>::const_iterator;

  // Special condition where
  // one rank only owns one tile
  // each tile is only owned by one rank
  bool rank_tile_bijection_ = false;
  bool has_owners_ = false;
  const vp_t local_rank_;
  const vp_t num_processes_;

  // Iterator to set of locally owned tiles
  TileSet locally_owned_tiles_;
  // Iterator to local rank MPI neighborhood
  RankSet local_rank_neighbors_;
  // Tile to owning ranks ( one-to-one mapping on number of tiles in grid )
  std::vector<std::set<vp_t>> tile_ranks_ownership_map_;
  // Rank to owned tiles ( one-to-one mapping on number of ranks in world )
  std::vector<std::set<tileidx_t>> rank_tiles_ownership_map_;
  // Rank to rank neighborhood ( one-to-one mapping on number of ranks in world
  // ) ( based on owned tile neighborhood )
  std::vector<std::pair<std::set<vp_t>, std::set<vp_t>>> rank_neighbors_map_;

  GridNeighborhood()
      : GridNeighborhood(get_mpi_rank(), get_num_mpi_processes()) {}

  GridNeighborhood(const GridNeighborhood &) = delete;
  GridNeighborhood(GridNeighborhood &&) noexcept = default;
  ~GridNeighborhood() noexcept = default;

  // The local rank selects this rank's ownership and neighbor entries
  GridNeighborhood(const vp_t local_rank, const vp_t num_processes)
      : local_rank_(local_rank), num_processes_(num_processes) {
    check_rank_in_range(local_rank, num_processes);
  }

  GridNeighborhood &operator=(const GridNeighborhood &) = delete;
  GridNeighborhood &operator=(GridNeighborhood &&) = delete;

  template <Coordinate CoordT>
  void set_tile_ownership(
      std::vector<std::set<tileidx_t>> &&rank_tiles_ownership_map,
      const TileGrid<CoordT> &tile_grid);

  const std::set<vp_t> &get_rank_neighborhood(const bool edge_wrap) const;

  bool in_neighborhood(const vp_t rank, const bool edge_wrap) const;

  // Whether some rank owns the tile (a valid tile index) or every tile
  bool tile_has_owner(const tileidx_t tile_index) const;
  bool all_tiles_owned() const;

  std::string to_string() const;
};

inline const std::set<vp_t> &
GridNeighborhood::get_rank_neighborhood(const bool edge_wrap) const {
  assert(has_owners_);
  return edge_wrap ? local_rank_neighbors_->second
                   : local_rank_neighbors_->first;
}

inline bool GridNeighborhood::in_neighborhood(const vp_t rank,
                                              const bool edge_wrap) const {
  assert(has_owners_);
  if (edge_wrap)
    return local_rank_neighbors_->second.contains(rank);
  else
    return local_rank_neighbors_->first.contains(rank);
}

inline bool GridNeighborhood::tile_has_owner(const tileidx_t tile_index) const {
  assert(has_owners_ && 0 <= tile_index &&
         std::cmp_less(tile_index, tile_ranks_ownership_map_.size()));
  return !tile_ranks_ownership_map_[static_cast<std::size_t>(tile_index)]
              .empty();
}

inline bool GridNeighborhood::all_tiles_owned() const {
  assert(has_owners_);
  return std::ranges::none_of(
      tile_ranks_ownership_map_,
      [](const auto &owners) { return owners.empty(); });
}

template <Coordinate CoordT>
void GridNeighborhood::set_tile_ownership(
    std::vector<std::set<tileidx_t>> &&rank_tiles_ownership_map,
    const TileGrid<CoordT> &tile_grid) {
  assert(!has_owners_ && !tile_grid.positions_.empty());

  if (!std::cmp_equal(rank_tiles_ownership_map.size(), num_processes_))
    throw std::invalid_argument("Ownership has to be defined for all ranks");
  // Checked before any state changes, so a rejected ownership leaves the
  // neighborhood as it was; every rank holds the same ownership and throws
  for (const auto &tile_set : rank_tiles_ownership_map)
    for (const auto &tile_idx : tile_set)
      if (tile_idx < 0 || tile_grid.num_tiles_ <= tile_idx)
        throw std::invalid_argument("Invalid tile index " +
                                    std::to_string(tile_idx) +
                                    " in tile ownership");

  vp_t curr_rank = 0;
  vp_t injective_ranks = 0;
  assert(0 <= tile_grid.num_tiles_);
  tile_ranks_ownership_map_.resize(
      static_cast<std::size_t>(tile_grid.num_tiles_));
  for (const auto &tile_set : rank_tiles_ownership_map) {
    injective_ranks += tile_set.size() == 1;
    for (const auto &tile_idx : tile_set)
      tile_ranks_ownership_map_[static_cast<std::size_t>(tile_idx)].insert(
          curr_rank);
    ++curr_rank;
  }

  tileidx_t injective_tiles = 0;
  auto tile_pos_it = tile_grid.positions_.cbegin();
  assert(0 <= num_processes_);
  rank_neighbors_map_.resize(static_cast<std::size_t>(num_processes_));
  for (const auto &source_rank_set : tile_ranks_ownership_map_) {
    assert(!tile_pos_it->direct_tile_neighborhood_.empty() &&
           !tile_pos_it->wrapped_tile_neighborhood_.empty());

    injective_tiles += source_rank_set.size() == 1;
    for (const auto &source_rank : source_rank_set) {
      const auto neighborhood_it = rank_neighbors_map_.begin() + source_rank;
      for (const auto &neighbor_index : tile_pos_it->direct_tile_neighborhood_)
        for (const auto &target_rank :
             tile_ranks_ownership_map_[static_cast<std::size_t>(
                 neighbor_index)])
          neighborhood_it->first.insert(target_rank);
      for (const auto &neighbor_index : tile_pos_it->wrapped_tile_neighborhood_)
        for (const auto &target_rank :
             tile_ranks_ownership_map_[static_cast<std::size_t>(
                 neighbor_index)])
          neighborhood_it->second.insert(target_rank);
    }

    ++tile_pos_it;
  }

  rank_tile_bijection_ = injective_ranks == num_processes_ &&
                         injective_tiles == tile_grid.num_tiles_;
  rank_tiles_ownership_map_ = std::move(rank_tiles_ownership_map);
  locally_owned_tiles_ = rank_tiles_ownership_map_.cbegin() + local_rank_;
  local_rank_neighbors_ = rank_neighbors_map_.cbegin() + local_rank_;
  has_owners_ = true;
}
} // namespace sapi

#endif
