/*
 *  node_containers_init.h
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

#ifndef NODE_CONTAINERS_INIT_H
#define NODE_CONTAINERS_INIT_H

#include <cassert>
#include <cstdint>
#include <set>
#include <vector>

#include "node_containers.h"
#include "sapi_forward_declarations.h"
#include "tile.h"
#include "tile_grid.h"

namespace sapi {
template <Coordinate CoordT>
inline void wrapped_split_tile(Tile<CoordT> &tile, const split_t num_splits,
                               const bool generate_total_leaves_vector) {
#pragma omp task default(none) shared(tile)                                    \
    firstprivate(num_splits, generate_total_leaves_vector)
  tile.split(num_splits, generate_total_leaves_vector);
}

template <Coordinate CoordT>
void wrapped_split_position(TilePosition<CoordT> &position,
                            const split_t num_splits,
                            const bool generate_total_leaves_vector) {
  wrapped_split_tile(position.tile_, num_splits, generate_total_leaves_vector);
  for (auto &image : position.tile_images_)
    wrapped_split_tile(image, num_splits, false);
}

template <Coordinate CoordT>
void split_tiles_in_grid(TileGrid<CoordT> &tile_grid,
                         const std::set<tileidx_t> &locally_owned_tiles,
                         const split_t num_splits) {
  static_assert(std::is_unsigned_v<split_t>);
  assert(!tile_grid.positions_.empty() && !tile_grid.has_split_);

  // A non-owned tile can neighbor several owned tiles, so mark every position
  // once before spawning tasks: each tile must be split by exactly one task.
  enum class SplitRole : uint8_t { NONE, NEIGHBOR, OWNED };
  std::vector<SplitRole> split_roles(tile_grid.positions_.size(),
                                     SplitRole::NONE);
  const auto role_of = [&split_roles](const tileidx_t tile_index) -> auto & {
    assert(0 <= tile_index);
    return split_roles.at(static_cast<std::size_t>(tile_index));
  };
  for (const auto &owned_tile : locally_owned_tiles)
    role_of(owned_tile) = SplitRole::OWNED;
  for (const auto &owned_tile : locally_owned_tiles)
    for (const auto &neighbor_index :
         tile_grid.positions_.at(static_cast<std::size_t>(owned_tile))
             .wrapped_tile_neighborhood_)
      if (role_of(neighbor_index) == SplitRole::NONE)
        role_of(neighbor_index) = SplitRole::NEIGHBOR;

#pragma omp parallel default(none) shared(tile_grid, split_roles)              \
    firstprivate(num_splits)
#pragma omp master
#pragma omp taskgroup
  for (std::size_t index = 0; index < split_roles.size(); ++index)
    if (split_roles[index] != SplitRole::NONE)
      wrapped_split_position(tile_grid.positions_[index], num_splits,
                             split_roles[index] == SplitRole::OWNED);

  tile_grid.has_split_ = true;
  tile_grid.splits_ = num_splits;
}

template <Coordinate CoordT>
void initialize_local_grid_node_collection(
    GridNodeCollection<CoordT> &grid_collection,
    const std::set<tileidx_t> &locally_owned_tiles,
    const TileGrid<CoordT> &tile_grid) {
  assert(grid_collection.empty() && tile_grid.has_split_);

  if (locally_owned_tiles.empty())
    return;

  const auto num_leaves = tile_grid.positions_.at(*locally_owned_tiles.cbegin())
                              .tile_.leaf_tiles_.size();
  assert(0 < num_leaves);

  grid_collection.resize(tile_grid.num_tiles_);
  const auto gc_it = grid_collection.begin();
  for (const auto &tile_pos : locally_owned_tiles)
    (gc_it + tile_pos)->resize(num_leaves);
}
} // namespace sapi

#endif
