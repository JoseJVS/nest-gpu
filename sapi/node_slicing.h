/*
 *  node_slicing.h
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

#ifndef NODE_SLICING_H
#define NODE_SLICING_H

#include <cassert>
#include <utility>

#include "grid_neighborhood.h"
#include "node_containers.h"
#include "sapi_forward_declarations.h"

namespace sapi {
template <Coordinate CoordT>
using FilteringCoordQueue = std::deque<IndexedCoordView<CoordT>>;

template <Coordinate CoordT>
std::deque<const Tile<CoordT> *> inline get_optional_masked_sub_tiles(
    const Tile<CoordT> &tile,
    const MaskCollection<CoordT> *const mask_collection) {
  std::deque<const Tile<CoordT> *> leaf_sub_tiles;

  if (mask_collection->has_source_mask())
    leaf_sub_tiles = mask_collection->source_overlapping_leafs(tile);
  else
    tile.insert_leaf_sub_tiles(leaf_sub_tiles);

  return leaf_sub_tiles;
}

// Without filtering every node of the leaf is checked and
// filtering_sequence is ignored
template <Coordinate CoordT, bool ignore_mask, bool filtered>
void check_insert_node_coord_pairs(
    FilteringCoordQueue<CoordT> &node_coord_pairs,
    const IndexedCoordCollection<CoordT> &indexed_coord_col,
    [[maybe_unused]] const NodeSequence &filtering_sequence,
    const Mask<CoordT> &mask) {
  static_assert(std::is_trivially_copyable_v<IndexedCoordView<CoordT>>);
  assert(node_coord_pairs.empty() && !indexed_coord_col.empty());

  if constexpr (filtered) {
    for (const auto &coord_vec : indexed_coord_col) {
      assert(!coord_vec.empty());

      // Taking into account the fact that vectors within
      // the coord_vec are sorted by coord index and contiguous
      // a quick jointure computation can be performed
      const NodeSequence original_sequence{
          coord_vec.cbegin()->first, static_cast<nodeidx_t>(coord_vec.size())};

      const auto jointure =
          join_sequences(original_sequence, filtering_sequence);

      if (jointure.second < 1)
        continue;

      const auto skip = jointure.first - original_sequence.first;
      assert(0 <= skip &&
             std::cmp_less_equal(skip + jointure.second, coord_vec.size()));
      const auto coord_it = coord_vec.begin() + skip;

      if constexpr (ignore_mask) {
        for (nodeidx_t index = 0; index < jointure.second; ++index)
          node_coord_pairs.emplace_back(coord_it + index);
      } else {
        for (nodeidx_t index = 0; index < jointure.second; ++index) {
          if (mask.coord_in_mask((coord_it + index)->second).first)
            node_coord_pairs.emplace_back(coord_it + index);
        }
      }
    }
  } else {
    for (const auto &coord_vec : indexed_coord_col) {
      assert(!coord_vec.empty());

      if constexpr (ignore_mask) {
        for (auto coord_it = coord_vec.cbegin(); coord_it != coord_vec.cend();
             ++coord_it)
          node_coord_pairs.emplace_back(coord_it);
      } else {
        for (auto coord_it = coord_vec.cbegin(); coord_it != coord_vec.cend();
             ++coord_it)
          if (mask.coord_in_mask(coord_it->second).first)
            node_coord_pairs.emplace_back(coord_it);
      }
    }
  }
}

// Appends one queue to tiled_coords per non-empty leaf of the tile that
// overlaps the source mask, and fills each queue in its own task
template <Coordinate CoordT, bool filtered>
void slice_tile(std::deque<FilteringCoordQueue<CoordT>> &tiled_coords,
                const tileidx_t tile_index,
                const NodeSequence filtering_sequence,
                const TileGrid<CoordT> &tile_grid,
                const GridNodeCollection<CoordT> &grid_node_col,
                const TAArray<MaskCollection<CoordT>> &mc_array) {
  const auto tile_nc_it = grid_node_col.cbegin() + tile_index;
  const auto tile_pos_it = tile_grid.positions_.cbegin() + tile_index;
  assert(tile_nc_it != grid_node_col.cend() &&
         tile_pos_it != tile_grid.positions_.cend());

  for (const auto &leaf_ptr : get_optional_masked_sub_tiles(
           tile_pos_it->tile_, mc_array.get_local_thread_item())) {
    const auto coord_map_it = tile_nc_it->cbegin() + leaf_ptr->index_;
    assert(coord_map_it != tile_nc_it->cend());

    if (coord_map_it->empty())
      continue;

    const auto leaf_coords = &tiled_coords.emplace_back();

#pragma omp task default(none) shared(mc_array)                                \
    firstprivate(leaf_coords, filtering_sequence, coord_map_it)
    {
      const auto mask_collection = mc_array.get_local_thread_item();
      if (mask_collection->has_source_mask())
        check_insert_node_coord_pairs<CoordT, false, filtered>(
            *leaf_coords, *coord_map_it, filtering_sequence,
            mask_collection->source_mask_);
      else
        check_insert_node_coord_pairs<CoordT, true, filtered>(
            *leaf_coords, *coord_map_it, filtering_sequence,
            mask_collection->source_mask_);
    }
  }
}

template <Coordinate CoordT>
std::deque<FilteringCoordQueue<CoordT>>
slice_tiled_node_maps(const DistributedTiledNodeSequenceMap *const dist_tns,
                      const TileGrid<CoordT> &tile_grid,
                      const GridNeighborhood &grid_neighborhood,
                      const GridNodeCollection<CoordT> &grid_node_col,
                      const TAArray<MaskCollection<CoordT>> &mc_array) {
  assert(tile_grid.has_split_ && grid_neighborhood.has_owners_ &&
         mc_array.is_initialized());

  std::deque<FilteringCoordQueue<CoordT>> tiled_coords;
  // A rank without tiles has no nodes to slice
  if (grid_node_col.empty())
    return tiled_coords;
  // Without a filter every locally owned tile is sliced in full
  const TileIdxNodeSequenceMap *local_tns = nullptr;
  if (dist_tns != nullptr) {
    // Also covers an empty population (no rank has nodes)
    auto local_tile_seq_map_it = dist_tns->find(grid_neighborhood.local_rank_);
    if (local_tile_seq_map_it == dist_tns->end())
      return tiled_coords;

    local_tns = &local_tile_seq_map_it->second;
  }

#pragma omp parallel default(none)                                             \
    shared(tile_grid, grid_neighborhood, grid_node_col, mc_array,              \
               tiled_coords, local_tns)
#pragma omp master
#pragma omp taskgroup
  {
    if (local_tns != nullptr)
      for (const auto &[tile_index, sequence] : *local_tns)
        slice_tile<CoordT, true>(tiled_coords, tile_index, sequence, tile_grid,
                                 grid_node_col, mc_array);
    else
      for (const auto tile_index : *grid_neighborhood.locally_owned_tiles_)
        slice_tile<CoordT, false>(tiled_coords, tile_index, NodeSequence{},
                                  tile_grid, grid_node_col, mc_array);
  }

  return tiled_coords;
}
} // namespace sapi

#endif
