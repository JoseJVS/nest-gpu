/*
 *  node_creation.h
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

#ifndef NODE_CREATION_H
#define NODE_CREATION_H

#include <algorithm>
#include <cassert>
#include <concepts>
#include <limits>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "dispatch.h"
#include "grid_neighborhood.h"
#include "node_containers.h"
#include "node_distribution.h"
#include "sapi_forward_declarations.h"
#include "task_helpers.h"
#include "tile.h"

namespace sapi {
// One leaf's share of a tile's node sequence
template <Coordinate CoordT> struct LeafCreationJob {
  IndexedCoordCollection<CoordT> *leaf_nodes_;
  const Tile<CoordT> *leaf_;
  nodeidx_t first_index_;
  nodeidx_t node_count_;
  tileidx_t tile_index_;
};

template <typename I>
  requires std::integral<std::remove_cvref_t<I>>
constexpr DISTRIBUTION_MODE get_distribution_mode(I &&mode) {
  switch (mode) {
  case uint8_t(DISTRIBUTION_MODE::FREE):
    return DISTRIBUTION_MODE::FREE;
  case uint8_t(DISTRIBUTION_MODE::SQUEEZED):
    return DISTRIBUTION_MODE::SQUEEZED;
  case uint8_t(DISTRIBUTION_MODE::BALANCED):
    return DISTRIBUTION_MODE::BALANCED;
  default:
    throw std::invalid_argument("Invalid distribution mode");
  }
}

template <bool balanced>
void aggregate_tiled_node_count_by_rank(
    NodeCountVector &node_counts_per_rank,
    RankTileIdxNodeCountPairs &tiled_node_counts_per_rank,
    const tileidx_t tile_index, const nodeidx_t node_count_in_tile,
    const GridNeighborhood &grid_neighborhood,
    const RandomManager &rng_manager) {
  assert(0 <= node_count_in_tile);

  if (node_count_in_tile == 0)
    return;

  const auto tile_owners =
      grid_neighborhood.tile_ranks_ownership_map_.cbegin() + tile_index;
  assert(tile_owners != grid_neighborhood.tile_ranks_ownership_map_.end());
  // Target tiles are checked for owners in distribute_node_counts_in_grid
  assert(!tile_owners->empty());

  const auto node_counts_per_owning_rank =
      uniform_distribute_node_counts<nodeidx_t, balanced, true>(
          node_count_in_tile, static_cast<tileidx_t>(tile_owners->size()),
          rng_manager);
  assert(node_counts_per_owning_rank.size() == tile_owners->size());

  auto nc_it = node_counts_per_owning_rank.cbegin();
  for (const auto &owner_rank : *tile_owners) {
    const auto count = *nc_it++;
    if (count == 0)
      continue;
    node_counts_per_rank[owner_rank] += count;
    tiled_node_counts_per_rank[owner_rank].emplace_back(
        std::make_pair(tile_index, count));
  }
}

// Aggregates the node counts of the tiles by owning rank; node_counts_per_tile
// holds one count per tile index, in the order of tile_indexes (ascending, so
// the balanced draws keep their order)
template <bool balanced, std::ranges::input_range R>
  requires std::convertible_to<std::ranges::range_value_t<R>, tileidx_t>
void aggregate_tile_node_counts(
    R &&tile_indexes, const NodeCountVector &node_counts_per_tile,
    NodeCountVector &node_counts_per_rank,
    RankTileIdxNodeCountPairs &tiled_node_counts_per_rank,
    const GridNeighborhood &grid_neighborhood,
    const RandomManager &rng_manager) {
  auto nc_it = node_counts_per_tile.cbegin();
  for (const tileidx_t tile_index : tile_indexes) {
    assert(nc_it != node_counts_per_tile.cend());
    aggregate_tiled_node_count_by_rank<balanced>(
        node_counts_per_rank, tiled_node_counts_per_rank, tile_index, *nc_it++,
        grid_neighborhood, rng_manager);
  }
  assert(nc_it == node_counts_per_tile.cend());
}

// Checks the arguments of distribute_node_counts_in_grid without drawing
// random numbers. Uses only data every rank holds, so all ranks throw the same
// std::invalid_argument.
template <typename I, Coordinate CoordT>
  requires std::integral<std::remove_cvref_t<I>>
void validate_node_distribution(
    const I num_nodes, const std::optional<std::set<tileidx_t>> &tile_set,
    const TileGrid<CoordT> &tile_grid,
    const GridNeighborhood &grid_neighborhood, const DISTRIBUTION_MODE mode) {
  const auto large_distribution =
      std::numeric_limits<nodeidx_t>::max() < num_nodes;

  if (num_nodes < 0)
    throw std::invalid_argument("Cannot distribute a negative number of nodes");
  if (std::numeric_limits<largenodeidx_t>::max() < num_nodes)
    throw std::invalid_argument(
        "Cannot distribute more than " +
        std::to_string(std::numeric_limits<largenodeidx_t>::max()) + " nodes");
  if (large_distribution && tile_set.has_value())
    throw std::invalid_argument("Distribution of a large number of nodes can "
                                "only be done over all tiles");
  if (large_distribution && !grid_neighborhood.rank_tile_bijection_)
    throw std::invalid_argument("Distribution of a large number of nodes can "
                                "only be done over a bijected grid");
  if (large_distribution && mode != DISTRIBUTION_MODE::BALANCED)
    throw std::invalid_argument(
        "Distribution of a large number of nodes can only be done using "
        "balanced distribution mode");
  if (tile_set.has_value() && mode == DISTRIBUTION_MODE::FREE)
    throw std::invalid_argument("Tile specific distribution of nodes cannot be "
                                "done using free distribution mode");

  if (tile_set.has_value() && (tile_set->empty() || *tile_set->cbegin() < 0 ||
                               tile_grid.num_tiles_ <= *tile_set->crbegin()))
    throw std::invalid_argument("Target tiles must be a non-empty set of "
                                "tile indexes of the grid");

  // Nodes can only be placed in tiles that some rank owns
  if (tile_set.has_value()
          ? !std::ranges::all_of(*tile_set,
                                 [&grid_neighborhood](const tileidx_t tile) {
                                   return grid_neighborhood.tile_has_owner(
                                       tile);
                                 })
          : !grid_neighborhood.all_tiles_owned())
    throw std::invalid_argument(
        "Nodes can only be distributed over tiles owned by some rank");

  // A balanced distribution gives each tile at most ceil(nodes / tiles)
  if (large_distribution &&
      std::cmp_less(std::numeric_limits<nodeidx_t>::max(),
                    num_nodes / tile_grid.num_tiles_ +
                        (num_nodes % tile_grid.num_tiles_ != 0)))
    throw std::invalid_argument(
        "Distribution of a large number of nodes needs more tiles: at most " +
        std::to_string(std::numeric_limits<nodeidx_t>::max()) +
        " nodes per tile");
}

// Here both returned vectors are in size == num processes.
// The first vector contains the total number of nodes
// to be instantiated in each rank.
// The second vector contains the total number of nodes
// to be instantiated per tile owned by each rank.
template <typename I, Coordinate CoordT>
  requires std::integral<std::remove_cvref_t<I>>
std::pair<NodeCountVector, RankTileIdxNodeCountPairs>
distribute_node_counts_in_grid(
    const I num_nodes, const std::optional<std::set<tileidx_t>> &tile_set,
    const TileGrid<CoordT> &tile_grid,
    const GridNeighborhood &grid_neighborhood, const RandomManager &rng_manager,
    const DISTRIBUTION_MODE mode) {
  assert(tile_grid.has_split_ && grid_neighborhood.has_owners_ &&
         rng_manager.is_initialized());

  validate_node_distribution(num_nodes, tile_set, tile_grid, grid_neighborhood,
                             mode);

  const auto large_distribution =
      std::numeric_limits<nodeidx_t>::max() < num_nodes;

  NodeCountVector node_counts_per_rank(grid_neighborhood.num_processes_, 0);
  RankTileIdxNodeCountPairs tiled_node_counts_per_rank(
      grid_neighborhood.num_processes_);

  if (num_nodes == 0)
    return std::make_pair(std::move(node_counts_per_rank),
                          std::move(tiled_node_counts_per_rank));

#pragma omp parallel default(none)                                             \
    shared(tile_set, tile_grid, grid_neighborhood, rng_manager,                \
               node_counts_per_rank, tiled_node_counts_per_rank)               \
    firstprivate(num_nodes, large_distribution, mode)
#pragma omp master
#pragma omp taskgroup
  {
    NodeCountVector node_counts_per_tile;

    switch (mode) {
    case DISTRIBUTION_MODE::FREE: {
      node_counts_per_tile =
          uniform_distribute_node_counts<nodeidx_t, GridPosition<CoordT>, true>(
              num_nodes, tile_grid.num_tiles_, tile_grid.dimensions_,
              rng_manager);

      break;
    }

    case DISTRIBUTION_MODE::SQUEEZED: {
      node_counts_per_tile =
          uniform_distribute_node_counts<nodeidx_t, false, true>(
              num_nodes,
              tile_set.has_value() ? static_cast<tileidx_t>(tile_set->size())
                                   : tile_grid.num_tiles_,
              rng_manager);

      break;
    }

    default: {
      if (large_distribution)
        node_counts_per_tile =
            uniform_distribute_node_counts<largenodeidx_t, true, true>(
                num_nodes, tile_grid.num_tiles_, rng_manager);
      else
        node_counts_per_tile =
            uniform_distribute_node_counts<nodeidx_t, true, true>(
                num_nodes,
                tile_set.has_value() ? static_cast<tileidx_t>(tile_set->size())
                                     : tile_grid.num_tiles_,
                rng_manager);

      break;
    }
    }

    assert(tile_set.has_value()
               ? node_counts_per_tile.size() == tile_set->size()
               : std::cmp_equal(node_counts_per_tile.size(),
                                tile_grid.num_tiles_));

    if (grid_neighborhood.rank_tile_bijection_ && !tile_set.has_value()) {
      for (vp_t rank = 0; rank < grid_neighborhood.num_processes_; ++rank) {
        const auto owned_tile =
            *grid_neighborhood.rank_tiles_ownership_map_[rank].cbegin();
        const auto tile_counts = node_counts_per_tile[owned_tile];
        node_counts_per_rank[rank] = tile_counts;
        tiled_node_counts_per_rank[rank].emplace_front(owned_tile, tile_counts);
      }
    } else {
      dispatch_bools(
          [&]<bool balanced>() {
            if (tile_set.has_value())
              aggregate_tile_node_counts<balanced>(
                  *tile_set, node_counts_per_tile, node_counts_per_rank,
                  tiled_node_counts_per_rank, grid_neighborhood, rng_manager);
            else
              aggregate_tile_node_counts<balanced>(
                  std::views::iota(tileidx_t{0}, tile_grid.num_tiles_),
                  node_counts_per_tile, node_counts_per_rank,
                  tiled_node_counts_per_rank, grid_neighborhood, rng_manager);
          },
          mode == DISTRIBUTION_MODE::BALANCED);
    }
  }

  return std::make_pair(std::move(node_counts_per_rank),
                        std::move(tiled_node_counts_per_rank));
}

// Generates the nodes of the given jobs. Every leaf reseeds the thread's RNG
// from its tile and leaf, so the output does not depend on how the jobs are
// grouped into tasks.
template <Coordinate CoordT>
inline void
generate_leaf_jobs(const std::span<const LeafCreationJob<CoordT>> jobs,
                   const RandomManager &rng_manager) {
  for (const auto &job : jobs) {
    assert(0 <= job.tile_index_ && 0 <= job.leaf_->index_);
    job.leaf_nodes_->emplace_back(job.leaf_->generate_coords_in_tile(
        job.first_index_, job.node_count_,
        *rng_manager.reseed_rank_paired_rng(
            static_cast<rng_seed_t>(job.tile_index_),
            static_cast<rng_seed_t>(job.leaf_->index_))));
  }
}

// Generates the jobs from first_job to the end of jobs in its own task
template <Coordinate CoordT>
inline void
spawn_leaf_job_chunk(const std::vector<LeafCreationJob<CoordT>> &jobs,
                     const nodeidx_t first_job,
                     const RandomManager &rng_manager) {
  assert(0 <= first_job && std::cmp_less(first_job, jobs.size()));
  const std::span<const LeafCreationJob<CoordT>> chunk(
      jobs.data() + first_job,
      jobs.size() - static_cast<std::size_t>(first_job));
#pragma omp task default(none) shared(rng_manager) firstprivate(chunk)
  generate_leaf_jobs(chunk, rng_manager);
}

template <Coordinate CoordT, bool use_branches, bool balanced>
void distribute_node_counts_in_tiles(
    GridNodeCollection<CoordT> &grid_node_col,
    const TileIdxNodeSequenceMap &node_seq_per_tile,
    const TileGrid<CoordT> &tile_grid, const RandomManager &rng_manager) {
  // Every tile of a grid has the grid's shape, so the split radices are the
  // same for all tiles.
  std::vector<split_t> possible_branches;
  if constexpr (use_branches)
    if (!node_seq_per_tile.empty())
      possible_branches =
          tile_grid
              .positions_[static_cast<std::size_t>(
                  node_seq_per_tile.cbegin()->first)]
              .tile_.get_possible_sub_tile_branches(tile_grid.splits_);

#pragma omp parallel default(none)                                             \
    shared(node_seq_per_tile, grid_node_col, tile_grid, rng_manager,           \
               possible_branches)
#pragma omp master
  {
    // One job per non-empty leaf, grouped into chunk tasks of similar node
    // count, since one task per leaf costs more than the generation itself
    // when leaves hold few nodes. Each chunk is spawned as soon as it is
    // complete, so generation overlaps with the node count distribution of
    // the following tiles. jobs is reserved for every leaf: it never
    // reallocates while running chunks read their jobs.
    nodeidx_t total_nodes = 0;
    nodeidx_t total_leaves = 0;
    for (const auto &[tile_index, node_sequence] : node_seq_per_tile) {
      const auto tile_leaves = (grid_node_col.cbegin() + tile_index)->size();
      assert(node_sequence.second <=
                 std::numeric_limits<nodeidx_t>::max() - total_nodes &&
             std::cmp_less_equal(tile_leaves,
                                 std::numeric_limits<nodeidx_t>::max() -
                                     total_leaves));
      total_nodes += node_sequence.second;
      total_leaves += static_cast<nodeidx_t>(tile_leaves);
    }
    const auto target = chunk_target(total_nodes, get_num_chunks());

    std::vector<LeafCreationJob<CoordT>> jobs;
    jobs.reserve(static_cast<std::size_t>(total_leaves));
    nodeidx_t chunk_first = 0;
    nodeidx_t chunk_nodes = 0;

#pragma omp taskgroup
    {
      for (const auto &[tile_index, node_sequence] : node_seq_per_tile) {
        assert(0 <= node_sequence.first && 0 < node_sequence.second);

        const auto tile_nc_it = grid_node_col.begin() + tile_index;
        const auto tile_pos_it = tile_grid.positions_.cbegin() + tile_index;
        assert(tile_nc_it != grid_node_col.end() && !tile_nc_it->empty() &&
               tile_pos_it != tile_grid.positions_.cend() &&
               tile_nc_it->size() == tile_pos_it->tile_.leaf_tiles_.size());
        assert(!use_branches ||
               possible_branches ==
                   tile_pos_it->tile_.get_possible_sub_tile_branches(
                       tile_grid.splits_));

        NodeCountVector node_count_per_sub_tile;

        if constexpr (use_branches) {
          node_count_per_sub_tile =
              uniform_distribute_node_counts<nodeidx_t, std::vector<split_t>,
                                             false>(
                  node_sequence.second,
                  static_cast<tileidx_t>(tile_nc_it->size()), possible_branches,
                  rng_manager);
        } else {
          node_count_per_sub_tile =
              uniform_distribute_node_counts<nodeidx_t, balanced, false>(
                  node_sequence.second,
                  static_cast<tileidx_t>(tile_nc_it->size()), rng_manager);
        }

        assert(node_count_per_sub_tile.size() == tile_nc_it->size());

        auto st_nc_it = tile_nc_it->begin();
        nodeidx_t st_node_index = node_sequence.first;
        auto leaf_it = tile_pos_it->tile_.leaf_tiles_.cbegin();
        for (const auto &node_count : node_count_per_sub_tile) {
          if (0 < node_count) {
            jobs.emplace_back(&*st_nc_it, *leaf_it, st_node_index, node_count,
                              tile_index);
            st_node_index += node_count;

            chunk_nodes += node_count;
            if (target <= chunk_nodes) {
              spawn_leaf_job_chunk(jobs, chunk_first, rng_manager);
              chunk_first = static_cast<nodeidx_t>(jobs.size());
              chunk_nodes = 0;
            }
          }

          ++leaf_it;
          ++st_nc_it;
        }
        assert((st_node_index - node_sequence.first) == node_sequence.second);
      }

      if (std::cmp_less(chunk_first, jobs.size()))
        spawn_leaf_job_chunk(jobs, chunk_first, rng_manager);
    }
  }
}

template <Coordinate CoordT>
void distribute_node_counts_in_tiles(
    GridNodeCollection<CoordT> &grid_node_col,
    const TileIdxNodeSequenceMap &node_seq_per_tile,
    const TileGrid<CoordT> &tile_grid, const RandomManager &rng_manager,
    const DISTRIBUTION_MODE mode) {
  assert(tile_grid.has_split_ && !grid_node_col.empty() &&
         rng_manager.is_initialized());

  if (node_seq_per_tile.empty())
    throw std::invalid_argument(
        "Cannot generate nodes with empty node sequences");

  switch (mode) {
  case DISTRIBUTION_MODE::FREE: {
    distribute_node_counts_in_tiles<CoordT, true, false>(
        grid_node_col, node_seq_per_tile, tile_grid, rng_manager);

    break;
  }

  case DISTRIBUTION_MODE::SQUEEZED: {
    distribute_node_counts_in_tiles<CoordT, false, false>(
        grid_node_col, node_seq_per_tile, tile_grid, rng_manager);

    break;
  }

  default: {
    distribute_node_counts_in_tiles<CoordT, false, true>(
        grid_node_col, node_seq_per_tile, tile_grid, rng_manager);

    break;
  }
  }

  rng_manager.update_rank_paired_seed(rng_manager.local_rank_);
}
} // namespace sapi

#endif
