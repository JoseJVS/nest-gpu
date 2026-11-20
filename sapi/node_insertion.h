/*
 *  node_insertion.h
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

#ifndef NODE_INSERTION_H
#define NODE_INSERTION_H

#include <cassert>
#include <tuple>
#include <utility>

#include "bounding_box.h"
#include "coordinate_geometry.h"
#include "grid_neighborhood.h"
#include "node_containers.h"
#include "node_distribution.h"
#include "sapi_forward_declarations.h"
#include "tile.h"

namespace sapi {
template <Coordinate CoordT>
bool recursive_bounds_test(const CoordT &coord,
                           TiledCoordMap<CoordT> &tile_coord_map,
                           const BoundingBox<CoordT> &bounding_box,
                           const TileGrid<CoordT> &tile_grid,
                           const GridNeighborhood &grid_neighborhood) {
  if (!bounding_box.coord_in_box(coord))
    return false;

  bool inserted = false;
  if (bounding_box.inner_boxes_.empty()) {
    assert(0 <= bounding_box.tile_index_ &&
           bounding_box.tile_index_ < tile_grid.num_tiles_);
    if (grid_neighborhood.tile_has_owner(bounding_box.tile_index_) &&
        tile_grid.positions_[bounding_box.tile_index_].tile_.coord_in_tile(
            coord)) {
      tile_coord_map[bounding_box.tile_index_].emplace_back(coord);
      inserted = true;
    }
  } else {
    for (const auto &bb : bounding_box.inner_boxes_) {
      inserted |= recursive_bounds_test(coord, tile_coord_map, bb, tile_grid,
                                        grid_neighborhood);
      if (inserted)
        break;
    }
  }

  return inserted;
}

template <Coordinate CoordT>
void aggregate_local_node_positions(std::deque<CoordT> &coords,
                                    TiledCoordMap<CoordT> &tiled_coord_map,
                                    const tileidx_t tile_index,
                                    const nodeidx_t copy_count,
                                    const nodeidx_t skip_count) {
  assert(0 < copy_count && 0 <= skip_count &&
         std::cmp_less_equal(skip_count + copy_count, coords.size()));

  // This function should only be called once per tile and only for local tiles
  if (0 == skip_count && std::cmp_equal(copy_count, coords.size())) {
    tiled_coord_map[tile_index].swap(coords);
    return;
  }

  const auto copy_begin = coords.cbegin() + skip_count;
  tiled_coord_map[tile_index].assign(copy_begin, copy_begin + copy_count);
  coords.clear();
}

template <Coordinate CoordT>
void aggregate_tiled_node_count_by_rank(
    TiledCoordMap<CoordT> &global_tiled_coord_map,
    TiledCoordMap<CoordT> &local_tiled_coord_map,
    NodeCountVector &node_counts_per_rank,
    RankTileIdxNodeCountPairs &tiled_node_counts_per_rank,
    const GridNeighborhood &grid_neighborhood,
    const RandomManager &rng_manager) {
  assert(std::cmp_equal(node_counts_per_rank.size(),
                        grid_neighborhood.num_processes_) &&
         std::cmp_equal(tiled_node_counts_per_rank.size(),
                        grid_neighborhood.num_processes_));

  if (global_tiled_coord_map.empty())
    return;

  for (auto &[tile_index, coords] : global_tiled_coord_map) {
    assert(!coords.empty() &&
           coords.size() < std::numeric_limits<nodeidx_t>::max());

    const auto tile_owners =
        grid_neighborhood.tile_ranks_ownership_map_.cbegin() + tile_index;
    assert(tile_owners != grid_neighborhood.tile_ranks_ownership_map_.end() &&
           !tile_owners->empty() &&
           tile_owners->size() < std::numeric_limits<tileidx_t>::max());

    const auto node_counts_per_owning_rank =
        uniform_distribute_node_counts<nodeidx_t, true, true>(
            static_cast<nodeidx_t>(coords.size()),
            static_cast<tileidx_t>(tile_owners->size()), rng_manager);

    nodeidx_t coord_skip = 0;
    auto nc_it = node_counts_per_owning_rank.cbegin();
    for (const auto &owner_rank : *tile_owners) {
      const auto count = *nc_it++;
      if (count == 0)
        continue;
      node_counts_per_rank[owner_rank] += count;
      tiled_node_counts_per_rank[owner_rank].emplace_back(tile_index, count);

      if (owner_rank != grid_neighborhood.local_rank_)
        coord_skip += count;
      else
        // As tile_owners is a set
        // this is guaranteed to happen only once per tile
        aggregate_local_node_positions(coords, local_tiled_coord_map,
                                       tile_index, count, coord_skip);
    }

    coords.clear();
  }

  global_tiled_coord_map.clear();
}

template <Coordinate CoordT>
std::tuple<NodeCountVector,           // rank node counts
           RankTileIdxNodeCountPairs, // rank tiled node counts
           TiledCoordMap<CoordT>      // sorted coords by locally owned tiles
           >
insert_node_positions_in_grid(std::deque<CoordT> &coords,
                              const TileGrid<CoordT> &tile_grid,
                              const GridNeighborhood &grid_neighborhood,
                              const RandomManager &rng_manager) {
  assert(tile_grid.has_split_ && grid_neighborhood.has_owners_ &&
         rng_manager.is_initialized());

  std::deque<CoordT> leftovers;
  NodeCountVector node_counts_per_rank(grid_neighborhood.num_processes_, 0);
  RankTileIdxNodeCountPairs tiled_node_counts_per_rank(
      grid_neighborhood.num_processes_);
  TiledCoordMap<CoordT> local_tiled_coords;

  if (coords.empty())
    return std::make_tuple(std::move(node_counts_per_rank),
                           std::move(tiled_node_counts_per_rank),
                           std::move(local_tiled_coords));

  TiledCoordMap<CoordT> global_tiled_coords;
  while (!coords.empty()) {
    if (!recursive_bounds_test(coords.front(), global_tiled_coords,
                               tile_grid.bounding_box_, tile_grid,
                               grid_neighborhood)) {
      leftovers.emplace_back(coords.front());
    }
    coords.pop_front();
  }

  aggregate_tiled_node_count_by_rank(
      global_tiled_coords, local_tiled_coords, node_counts_per_rank,
      tiled_node_counts_per_rank, grid_neighborhood, rng_manager);

  coords.swap(leftovers);

  return std::make_tuple(std::move(node_counts_per_rank),
                         std::move(tiled_node_counts_per_rank),
                         std::move(local_tiled_coords));
}

template <Coordinate CoordT>
void recursive_sub_tile_test(const CoordT &coord,
                             TiledCoordMap<CoordT> &sub_tile_coord_map,
                             const Tile<CoordT> &tile) {
  // Tile coord check is performed at previous recursive call
  // for root tile this is done during recursive bounds test
  if (tile.sub_tiles_.empty()) {
    sub_tile_coord_map[tile.index_].emplace_back(coord);
  } else {
    const Tile<CoordT> *sub_tile = nullptr;
    for (const auto &st : tile.sub_tiles_) {
      if (st.coord_in_tile(coord)) {
        sub_tile = &st;
        break;
      }
    }

    if (sub_tile != nullptr)
      recursive_sub_tile_test(coord, sub_tile_coord_map, *sub_tile);
    else {
      space_t min_distance = std::numeric_limits<space_t>::max();
      for (const auto &st : tile.sub_tiles_) {
        const auto distance =
            distance2(coord, st.project_point_to_surface(coord));

        if (std::isless(distance, min_distance)) {
          min_distance = distance;
          sub_tile = &st;
        }
      }

      assert(sub_tile != nullptr);
      recursive_sub_tile_test(coord, sub_tile_coord_map, *sub_tile);
    }
  }
}

template <Coordinate CoordT>
void insert_node_positions_in_leafs(std::deque<CoordT> &coords,
                                    LeafNodeCollection<CoordT> &leaf_node_col,
                                    const TilePosition<CoordT> &tile_position,
                                    const NodeSequence &node_sequence) {
  assert(!coords.empty());

  TiledCoordMap<CoordT> sub_tile_coord_map;
  for (const auto &coord : coords)
    recursive_sub_tile_test(coord, sub_tile_coord_map, tile_position.tile_);

  coords.clear();
  assert(!sub_tile_coord_map.empty());

  nodeidx_t st_node_index = node_sequence.first;
  for (auto &[st_index, st_coords] : sub_tile_coord_map) {
    assert(!st_coords.empty() &&
           st_coords.size() < std::numeric_limits<nodeidx_t>::max());

    const auto indexed_coord_col = leaf_node_col.begin() + st_index;
    assert(indexed_coord_col != leaf_node_col.end());

    auto &coord_vec = indexed_coord_col->emplace_back();
    coord_vec.reserve(st_coords.size());

    for (const auto &coord : st_coords)
      coord_vec.emplace_back(st_node_index++, coord);
    st_coords.clear();
  }
  assert((st_node_index - node_sequence.first) == node_sequence.second);

  sub_tile_coord_map.clear();
}

template <Coordinate CoordT>
void insert_node_positions_in_tiles(
    TiledCoordMap<CoordT> &tiled_coord_map,
    GridNodeCollection<CoordT> &grid_node_col,
    const TileIdxNodeSequenceMap &node_seq_per_tile,
    const TileGrid<CoordT> &tile_grid) {
  assert(tile_grid.has_split_ && !grid_node_col.empty());

  if (node_seq_per_tile.empty())
    throw std::invalid_argument(
        "Cannot generate nodes with empty node sequences");
  if (node_seq_per_tile.size() != tiled_coord_map.size())
    throw std::invalid_argument(
        "Tiled node sequence mismatch with tiled coord map");

  // Each tile reads and clears only its own coords and fills only its own
  // leaf collection, and its node indices come from its node sequence, so
  // the tiles are independent tasks.
#pragma omp parallel default(none)                                             \
    shared(tiled_coord_map, grid_node_col, node_seq_per_tile, tile_grid)
#pragma omp master
#pragma omp taskgroup
  for (const auto &[tile_index, node_sequence] : node_seq_per_tile) {
    const auto tcm_it = tiled_coord_map.find(tile_index);
    const auto tnc_it = grid_node_col.begin() + tile_index;
    assert(tcm_it != tiled_coord_map.end() && tnc_it != grid_node_col.end() &&
           0 <= node_sequence.first && 0 < node_sequence.second &&
           std::cmp_equal(tcm_it->second.size(), node_sequence.second));

#pragma omp task default(none) shared(tile_grid)                               \
    firstprivate(tcm_it, tnc_it, tile_index, node_sequence)
    insert_node_positions_in_leafs(tcm_it->second, *tnc_it,
                                   tile_grid.positions_[tile_index],
                                   node_sequence);
  }

  tiled_coord_map.clear();
}
} // namespace sapi

#endif
