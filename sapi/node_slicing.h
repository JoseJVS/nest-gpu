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
#include <cstddef>
#include <deque>
#include <limits>
#include <span>
#include <utility>
#include <vector>

#include "grid_neighborhood.h"
#include "node_containers.h"
#include "sapi_forward_declarations.h"
#include "task_helpers.h"

namespace sapi {
template <Coordinate CoordT>
using FilteringCoordQueue = std::deque<IndexedCoordView<CoordT>>;

// One non-empty leaf to slice into its own output queue
template <Coordinate CoordT> struct LeafSliceJob {
  FilteringCoordQueue<CoordT> *leaf_coords_;
  const IndexedCoordCollection<CoordT> *coords_;
  NodeSequence filtering_sequence_;
};

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

// Appends one queue to tiled_coords and one job per non-empty leaf of the
// tile that overlaps the source mask; job_weights gets the leaf's node count
template <Coordinate CoordT>
inline void collect_slice_jobs(
    std::deque<FilteringCoordQueue<CoordT>> &tiled_coords,
    std::vector<LeafSliceJob<CoordT>> &jobs,
    std::vector<nodeidx_t> &job_weights, const tileidx_t tile_index,
    const NodeSequence filtering_sequence, const TileGrid<CoordT> &tile_grid,
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

    nodeidx_t leaf_node_count = 0;
    for (const auto &coord_vec : *coord_map_it) {
      assert(std::cmp_less_equal(coord_vec.size(),
                                 std::numeric_limits<nodeidx_t>::max() -
                                     leaf_node_count));
      leaf_node_count += static_cast<nodeidx_t>(coord_vec.size());
    }

    jobs.emplace_back(&tiled_coords.emplace_back(), &*coord_map_it,
                      filtering_sequence);
    job_weights.push_back(leaf_node_count);
  }
}

// Slices the leaves of the given jobs, each into its own queue
template <Coordinate CoordT, bool filtered>
inline void slice_leaf_jobs(const std::span<const LeafSliceJob<CoordT>> jobs,
                            const TAArray<MaskCollection<CoordT>> &mc_array) {
  const auto mask_collection = mc_array.get_local_thread_item();
  for (const auto &job : jobs) {
    if (mask_collection->has_source_mask())
      check_insert_node_coord_pairs<CoordT, false, filtered>(
          *job.leaf_coords_, *job.coords_, job.filtering_sequence_,
          mask_collection->source_mask_);
    else
      check_insert_node_coord_pairs<CoordT, true, filtered>(
          *job.leaf_coords_, *job.coords_, job.filtering_sequence_,
          mask_collection->source_mask_);
  }
}

// Groups the slice jobs into chunk tasks of similar node count, since one
// task per leaf costs more than the slicing itself when leaves hold few nodes
template <Coordinate CoordT, bool filtered>
inline void
slice_jobs_in_chunks(const std::vector<LeafSliceJob<CoordT>> &jobs,
                     const std::vector<nodeidx_t> &job_weights,
                     const TAArray<MaskCollection<CoordT>> &mc_array) {
  if (jobs.empty())
    return;

  const auto bounds = chunk_bounds(job_weights, get_num_chunks());

#pragma omp taskgroup
  for (std::size_t chunk = 0; chunk + 1 < bounds.size(); ++chunk) {
    assert(0 <= bounds[chunk] && bounds[chunk] <= bounds[chunk + 1]);
    const std::span<const LeafSliceJob<CoordT>> chunk_jobs(
        jobs.data() + bounds[chunk],
        static_cast<std::size_t>(bounds[chunk + 1] - bounds[chunk]));
#pragma omp task default(none) shared(mc_array) firstprivate(chunk_jobs)
    slice_leaf_jobs<CoordT, filtered>(chunk_jobs, mc_array);
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
  {
    std::vector<LeafSliceJob<CoordT>> jobs;
    std::vector<nodeidx_t> job_weights;

    if (local_tns != nullptr) {
      for (const auto &[tile_index, sequence] : *local_tns)
        collect_slice_jobs(tiled_coords, jobs, job_weights, tile_index,
                           sequence, tile_grid, grid_node_col, mc_array);
      slice_jobs_in_chunks<CoordT, true>(jobs, job_weights, mc_array);
    } else {
      for (const auto tile_index : *grid_neighborhood.locally_owned_tiles_)
        collect_slice_jobs(tiled_coords, jobs, job_weights, tile_index,
                           NodeSequence{}, tile_grid, grid_node_col, mc_array);
      slice_jobs_in_chunks<CoordT, false>(jobs, job_weights, mc_array);
    }
  }

  return tiled_coords;
}
} // namespace sapi

#endif
