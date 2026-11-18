/*
 *  mask_tile_processing.h
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

#ifndef MASK_TILE_PROCESSING_H
#define MASK_TILE_PROCESSING_H

#include <cassert>
#include <cstdint>
#include <memory>
#include <utility>

#include "enum_store.h"
#include "grid_neighborhood.h"
#include "mask_containers.h"
#include "node_containers.h"
#include "payload_preparation.h"
#include "sapi_forward_declarations.h"

namespace sapi {
template <Coordinate CoordT, bool inverted_source_target, bool filter_source,
          bool filter_target>
inline bool no_overlap(const Tile<CoordT> &tile,
                       const MaskCollection<CoordT> *const mask_collection) {
  if constexpr (inverted_source_target) {
    if constexpr (filter_target) {
      return !mask_collection->target_overlap(tile);
    } else {
      return false;
    }
  } else {
    if constexpr (filter_source) {
      return !mask_collection->source_overlap(tile);
    } else {
      return false;
    }
  }
}

template <Coordinate CoordT, bool inverted_source_target, bool filter_source,
          bool filter_target>
void insert_leaf_tiles_within_range(
    std::unordered_map<tileidx_t, std::unordered_map<tileidx_t, count_t>>
        &leaf_pairs,
    const Tile<CoordT> &driver, const Tile<CoordT> &pool,
    const TAArray<MaskCollection<CoordT>> &mc_array, const shift_t image_index,
    const bool is_shifted, const split_t splits) {
  assert(driver.sub_tiles_.size() == pool.sub_tiles_.size());

  if (driver.sub_tiles_.empty()) {
    assert(splits == 0 && 0 <= pool.index_ && 0 <= driver.index_);

    if constexpr (inverted_source_target) {
#pragma omp critical(sapi_leaf_pairs)
      leaf_pairs[pool.index_][driver.index_] |= 1 << image_index;
    } else {
#pragma omp critical(sapi_leaf_pairs)
      leaf_pairs[driver.index_][pool.index_] |= 1 << image_index;
    }
  } else {
    assert(0 < splits);
    const auto d_st_count = driver.sub_tiles_.size();
    const auto d_sub_tiles = driver.sub_tiles_.data();
    const auto p_st_count = pool.sub_tiles_.size();
    const auto p_sub_tiles = pool.sub_tiles_.data();
    const auto rem_splits = splits - 1;

    // Sub-tiles passing their mask, one bit per position in sub_tiles_, so
    // each sub-tile is tested once instead of once per pair. The driver is
    // always the source tile and the pool the target tile, on both the source
    // and the (inverted) target side, see tile_pair_overlap callers. Without
    // source/target filtering no_overlap is false and the bits are all set.
    assert(d_st_count <= 64 && p_st_count <= 64);
    const auto mask_collection = mc_array.get_local_thread_item();
    uint64_t d_st_ok = 0, p_st_ok = 0;
    for (std::size_t st = 0; st < d_st_count; ++st)
      d_st_ok |= static_cast<uint64_t>(
                     !no_overlap<CoordT, false, filter_source, filter_target>(
                         d_sub_tiles[st], mask_collection))
                 << st;
    for (std::size_t st = 0; st < p_st_count; ++st)
      p_st_ok |= static_cast<uint64_t>(
                     is_shifted ||
                     !no_overlap<CoordT, true, filter_source, filter_target>(
                         p_sub_tiles[st], mask_collection))
                 << st;

    if (0 == d_st_ok || 0 == p_st_ok)
      return;

#pragma omp taskloop collapse(2) num_tasks(strict : 4) nogroup mergeable       \
final(rem_splits < 2) default(none) shared(leaf_pairs, mc_array)               \
    firstprivate(d_st_count, d_sub_tiles, p_st_count, p_sub_tiles,             \
                     image_index, is_shifted, rem_splits, d_st_ok, p_st_ok)
    for (std::size_t d_st = 0; d_st < d_st_count; ++d_st) {
      for (std::size_t p_st = 0; p_st < p_st_count; ++p_st) {
        if ((d_st_ok >> d_st & 1) && (p_st_ok >> p_st & 1) &&
            mc_array.get_local_thread_item()->blueprint_overlap(
                d_sub_tiles[d_st], p_sub_tiles[p_st])) {
          insert_leaf_tiles_within_range<CoordT, inverted_source_target,
                                         filter_source, filter_target>(
              leaf_pairs, d_sub_tiles[d_st], p_sub_tiles[p_st], mc_array,
              image_index, is_shifted, rem_splits);
        }
      }
    }
  }
}

template <Coordinate CoordT, bool edge_wrap, bool inverted_source_target,
          bool filter_source, bool filter_target>
void tile_pair_overlap(TilePairInfo<CoordT> &tpi,
                       const TilePosition<CoordT> &driver_tile_pos,
                       const TilePosition<CoordT> &pool_tile_pos,
                       const TAArray<MaskCollection<CoordT>> &mc_array,
                       const split_t splits) {
  assert(tpi.aggregated_leaf_pairs_.empty() &&
         !pool_tile_pos.tile_images_.empty() &&
         pool_tile_pos.image_displacements_.size() ==
             pool_tile_pos.tile_images_.size());

  const auto mask_collection = mc_array.get_local_thread_item();
  tpi.image_displacements_ = std::addressof(pool_tile_pos.image_displacements_);

#pragma omp taskgroup
  {
    OmpLockGuard<CoordT> g(driver_tile_pos.tile_lock_);
    driver_tile_pos.tile_.split(splits, false);
  }

  if constexpr (edge_wrap) {
    assert(pool_tile_pos.tile_images_.size() <= 27);
    const shift_t total_images =
        static_cast<shift_t>(pool_tile_pos.tile_images_.size());

    for (shift_t image_index = 0; image_index < total_images; ++image_index) {
      auto pool_image = &pool_tile_pos.tile_images_[image_index];
      auto image_lock = &pool_tile_pos.image_locks_[image_index];

      const bool is_shifted = pool_image->shape_ != TILE_SHAPE::NULL_TS;

      if (!is_shifted) {
        pool_image = &pool_tile_pos.tile_;
        image_lock = &pool_tile_pos.tile_lock_;
      }

      if (mask_collection->blueprint_overlap(driver_tile_pos.tile_,
                                             *pool_image)) {
#pragma omp taskgroup
        {
          OmpLockGuard g(*image_lock);
          pool_image->split(splits, false);
        }

#pragma omp task default(none) shared(tpi, driver_tile_pos, mc_array)          \
    firstprivate(pool_image, image_index, is_shifted, splits)
        insert_leaf_tiles_within_range<CoordT, inverted_source_target,
                                       filter_source, filter_target>(
            tpi.aggregated_leaf_pairs_, driver_tile_pos.tile_, *pool_image,
            mc_array, image_index, is_shifted, splits);
      }
    }
  } else {
    if (!mask_collection->blueprint_overlap(driver_tile_pos.tile_,
                                            pool_tile_pos.tile_))
      return;

#pragma omp taskgroup
    {
      OmpLockGuard<CoordT> g(pool_tile_pos.tile_lock_);
      pool_tile_pos.tile_.split(splits, false);
    }

    insert_leaf_tiles_within_range<CoordT, inverted_source_target,
                                   filter_source, filter_target>(
        tpi.aggregated_leaf_pairs_, driver_tile_pos.tile_, pool_tile_pos.tile_,
        mc_array,
        0, // the first image in the image vector is never shifted
        false, splits);
  }
}

template <Coordinate CoordT, bool ignore_mask>
void check_insert_node_coord_pairs(
    FilteredIndexedCoordinates<CoordT> &node_coord_pairs,
    const IndexedCoordCollection<CoordT> &indexed_coord_col,
    const Mask<CoordT> &mask, const NodeSequence &node_sequence) {
  static_assert(std::is_trivially_copyable_v<IndexedCoordView<CoordT>>);
  assert(node_coord_pairs.empty() && !indexed_coord_col.empty());

  nodeidx_t sequences_found = 0;
  for (const auto &coord_vec : indexed_coord_col) {
    assert(!coord_vec.empty());

    // Taking into account the fact that vectors within
    // the coord_vec are sorted by coord index and contiguous
    // a quick jointure computation can be performed
    const nodeidx_t first_index = coord_vec.cbegin()->first;
    const auto jointure = join_sequences(
        NodeSequence(first_index, static_cast<nodeidx_t>(coord_vec.size())),
        node_sequence);

    if (jointure.second < 1)
      continue;

    // Due to how sequences are stored in the distributed tiled node sequence
    // map there is no apriori of which subsequence is stored in each leaf
    // however during mask tile processing at most one subsequence should match
    ++sequences_found;
    node_coord_pairs.reserve(jointure.second);

    const auto skip = jointure.first - first_index;
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

#ifdef NDEBUG
    break;
#endif
  }

  assert(sequences_found <= 1);
}

template <Coordinate CoordT, bool inverted_source_target, bool filter_source,
          bool filter_target>
void aggregate_tile_set_info(
    TileSetInfo<CoordT> &tsi, FilteredTileNodeCollection<CoordT> &icp,
    const std::vector<std::pair<tileidx_t, TilePairInfo<CoordT>>> &tpi_vec,
    const LeafNodeCollection<CoordT> &leaf_node_col,
    const TAArray<MaskCollection<CoordT>> &mc_array,
    const NodeSequence &node_sequence, const tileidx_t source_tile_idx) {
  assert(!leaf_node_col.empty());

  // At this point the given source tile pos
  // was checked against all target tile pos
  // from a given target rank.
  // Hence, tpi_vec is complete and unique
  // to the source/target ranks pair.

  FilteredLeafNodeCollection<CoordT> checked_leaves;

  // At most all the leaves will be used
  checked_leaves.reserve(leaf_node_col.size());

  for (const auto &tpi_pair : tpi_vec)
    for (const auto &leaf_map : tpi_pair.second.aggregated_leaf_pairs_)
      checked_leaves.emplace(leaf_map.first,
                             FilteredIndexedCoordinates<CoordT>());

#pragma omp taskgroup
  for (auto leaf_it = checked_leaves.begin(); leaf_it != checked_leaves.end();
       ++leaf_it) {
    const auto ncv_it = leaf_node_col.cbegin() + leaf_it->first;
    assert(ncv_it != leaf_node_col.cend());

    if (ncv_it->empty())
      continue;

#pragma omp task default(none) shared(mc_array)                                \
    firstprivate(leaf_it, node_sequence, ncv_it)
    {
      if constexpr (inverted_source_target) {
        check_insert_node_coord_pairs<CoordT, !filter_target>(
            leaf_it->second, *ncv_it,
            mc_array.get_local_thread_item()->target_mask_, node_sequence);
      } else {
        check_insert_node_coord_pairs<CoordT, !filter_source>(
            leaf_it->second, *ncv_it,
            mc_array.get_local_thread_item()->source_mask_, node_sequence);
      }
    }
  }

  nodeidx_t total_nodes = 0;
  auto check_it = checked_leaves.cbegin();
  const auto check_end = checked_leaves.cend();
  while (check_it != check_end) {
    if (check_it->second.empty())
      check_it = checked_leaves.erase(check_it);
    else
      total_nodes += (*check_it++).second.size();
  }

  if (checked_leaves.empty())
    return;
  assert(0 < total_nodes);

#pragma omp atomic
  ++tsi.valid_tiles_;
#pragma omp atomic
  tsi.valid_leaves_ += checked_leaves.size();
#pragma omp atomic
  tsi.total_node_count_ += total_nodes;

#pragma omp critical(sapi_filtered_coords)
  {
    const auto emplace_res =
        icp.emplace(source_tile_idx, std::move(checked_leaves)).second;
    assert(emplace_res);
  }
}

template <Coordinate CoordT, bool edge_wrap, bool only_neighborhood,
          bool inverted_source_target, bool filter_source, bool filter_target>
void rank_pair_overlap(RankPairInfo<CoordT> &rpi,
                       const TileIdxNodeSequenceMap &tns_source,
                       const TileIdxNodeSequenceMap &tns_target,
                       const TileGrid<CoordT> &tile_grid,
                       const GridNodeCollection<CoordT> &grid_node_col,
                       const TAArray<MaskCollection<CoordT>> &mc_array,
                       const bool prepare_payload) {
  assert(rpi.tile_pairs_set_.tile_pairings_.empty() && !tns_target.empty());

  const auto mask_collection = mc_array.get_local_thread_item();

  rpi.tile_pairs_set_.tile_pairings_.reserve(tns_source.size());

#pragma omp taskgroup
  for (const auto &[source_tile_index, source_node_sequence] : tns_source) {
    const auto source_tile_pos =
        tile_grid.positions_.cbegin() + source_tile_index;

    if (no_overlap<CoordT, inverted_source_target, filter_source,
                   filter_target>(source_tile_pos->tile_, mask_collection))
      continue;

    bool valid_tiles = false;
    const auto source_pair = &rpi.tile_pairs_set_.tile_pairings_.emplace_back(
        source_tile_index,
        std::vector<std::pair<tileidx_t, TilePairInfo<CoordT>>>());
    source_pair->second.reserve(tns_target.size());

#pragma omp taskgroup
    for (const auto &[target_tile_index, target_node_sequence] : tns_target) {
      if constexpr (only_neighborhood) {
        if (!source_tile_pos->in_neighborhood(target_tile_index, edge_wrap))
          continue;
      }

      const auto target_tile_pos =
          tile_grid.positions_.cbegin() + target_tile_index;

      if (no_overlap<CoordT, !inverted_source_target, filter_source,
                     filter_target>(target_tile_pos->tile_, mask_collection))
        continue;

      const auto target_pair = &source_pair->second.emplace_back(
          target_tile_index, TilePairInfo<CoordT>());
      valid_tiles = true;

#pragma omp task default(none) shared(mc_array, tile_grid)                     \
    firstprivate(target_pair, source_tile_pos, target_tile_pos)
      {
        if constexpr (inverted_source_target) {
          tile_pair_overlap<CoordT, edge_wrap, inverted_source_target,
                            filter_source, filter_target>(
              target_pair->second, *target_tile_pos, *source_tile_pos, mc_array,
              tile_grid.splits_);
        } else {
          tile_pair_overlap<CoordT, edge_wrap, inverted_source_target,
                            filter_source, filter_target>(
              target_pair->second, *source_tile_pos, *target_tile_pos, mc_array,
              tile_grid.splits_);
        }
      }
    }

    if (valid_tiles) {
      const auto tile_nc_it = grid_node_col.begin() + source_tile_index;
      assert(tile_nc_it != grid_node_col.end());

#pragma omp task default(none) shared(rpi, mc_array)                           \
    firstprivate(source_pair, tile_nc_it, source_node_sequence)
      aggregate_tile_set_info<CoordT, inverted_source_target, filter_source,
                              filter_target>(
          rpi.tile_pairs_set_, rpi.sender_info_.filtered_coords_,
          source_pair->second, *tile_nc_it, mc_array, source_node_sequence,
          source_pair->first);
    }
  }

  if (prepare_payload)
    prepare_data_payload(rpi.sender_info_, rpi.tile_pairs_set_);
}

template <Coordinate CoordT, bool edge_wrap, bool only_neighborhood,
          bool inverted_source_target, bool filter_source, bool filter_target>
void distributed_overlap(
    std::unordered_map<vp_t, RankPairInfo<CoordT>> &rpi_map,
    const TileIdxNodeSequenceMap &tns_source,
    const DistributedTiledNodeSequenceMap &dist_tns_target,
    const TileGrid<CoordT> &tile_grid,
    const GridNeighborhood &grid_neighborhood,
    const GridNodeCollection<CoordT> &grid_node_col,
    const TAArray<MaskCollection<CoordT>> &mc_array,
    const bool force_prepare_payload) {
  assert(rpi_map.empty() && !tns_source.empty() && !grid_node_col.empty());

  for (auto tns_target_it = dist_tns_target.cbegin();
       tns_target_it != dist_tns_target.cend(); ++tns_target_it) {
    const auto is_remote =
        tns_target_it->first != grid_neighborhood.local_rank_;
    if constexpr (only_neighborhood) {
      if (is_remote &&
          !grid_neighborhood.in_neighborhood(tns_target_it->first, edge_wrap))
        continue;
    }

    const auto [emplace_it, success] =
        rpi_map.emplace(tns_target_it->first, RankPairInfo<CoordT>());
    assert(success);

#pragma omp task default(none)                                                 \
    shared(tns_source, tile_grid, grid_node_col, mc_array)                     \
    firstprivate(emplace_it, tns_target_it, force_prepare_payload, is_remote)
    rank_pair_overlap<CoordT, edge_wrap, only_neighborhood,
                      inverted_source_target, filter_source, filter_target>(
        emplace_it->second, tns_source, tns_target_it->second, tile_grid,
        grid_node_col, mc_array, force_prepare_payload || is_remote);
  }
}

template <Coordinate CoordT, bool edge_wrap, bool only_neighborhood>
std::pair<bool, bool> compute_distributed_tile_overlap(
    DistributedPairInfo<CoordT> &dpi,
    const DistributedTiledNodeSequenceMap &dist_tns_source,
    const DistributedTiledNodeSequenceMap &dist_tns_target,
    const TileGrid<CoordT> &tile_grid,
    const GridNeighborhood &grid_neighborhood,
    const GridNodeCollection<CoordT> &grid_node_col,
    const TAArray<MaskCollection<CoordT>> &mc_array,
    const bool force_prepare_payload = false) {
  assert(!dist_tns_source.empty() && !dist_tns_target.empty() &&
         tile_grid.has_split_ && grid_neighborhood.has_owners_ &&
         mc_array.is_initialized());

  bool is_source = false;
  bool is_target = false;
  const auto source_side = dist_tns_source.find(grid_neighborhood.local_rank_);
  const auto target_side = dist_tns_target.find(grid_neighborhood.local_rank_);

  const auto mask_collection = mc_array.get_local_thread_item();
  assert(mask_collection->has_blueprint());

  const bool filter_source = mask_collection->has_source_mask();
  const bool filter_target = mask_collection->has_target_mask();

#pragma omp taskgroup
  {
    if (source_side != dist_tns_source.end()) {
#pragma omp task default(none)                                                 \
    shared(dpi, dist_tns_target, tile_grid, grid_neighborhood, grid_node_col,  \
               mc_array) firstprivate(filter_source, filter_target,            \
                                          source_side, force_prepare_payload)
      switch ((filter_source << 0) + (filter_target << 1)) {
      case 1: {
        distributed_overlap<CoordT, edge_wrap, only_neighborhood, false, true,
                            false>(dpi.source_side_info_, source_side->second,
                                   dist_tns_target, tile_grid,
                                   grid_neighborhood, grid_node_col, mc_array,
                                   force_prepare_payload);

        break;
      }

      case 2: {
        distributed_overlap<CoordT, edge_wrap, only_neighborhood, false, false,
                            true>(dpi.source_side_info_, source_side->second,
                                  dist_tns_target, tile_grid, grid_neighborhood,
                                  grid_node_col, mc_array,
                                  force_prepare_payload);

        break;
      }

      case 3: {
        distributed_overlap<CoordT, edge_wrap, only_neighborhood, false, true,
                            true>(dpi.source_side_info_, source_side->second,
                                  dist_tns_target, tile_grid, grid_neighborhood,
                                  grid_node_col, mc_array,
                                  force_prepare_payload);

        break;
      }

      default: {
        distributed_overlap<CoordT, edge_wrap, only_neighborhood, false, false,
                            false>(dpi.source_side_info_, source_side->second,
                                   dist_tns_target, tile_grid,
                                   grid_neighborhood, grid_node_col, mc_array,
                                   force_prepare_payload);

        break;
      }
      }
    }

    if (target_side != dist_tns_target.end()) {
#pragma omp task default(none)                                                 \
    shared(dpi, dist_tns_source, tile_grid, grid_neighborhood, grid_node_col,  \
               mc_array) firstprivate(filter_source, filter_target,            \
                                          target_side, force_prepare_payload)
      switch ((filter_source << 0) + (filter_target << 1)) {
      case 1: {
        distributed_overlap<CoordT, edge_wrap, only_neighborhood, true, true,
                            false>(dpi.target_side_info_, target_side->second,
                                   dist_tns_source, tile_grid,
                                   grid_neighborhood, grid_node_col, mc_array,
                                   force_prepare_payload);

        break;
      }

      case 2: {
        distributed_overlap<CoordT, edge_wrap, only_neighborhood, true, false,
                            true>(dpi.target_side_info_, target_side->second,
                                  dist_tns_source, tile_grid, grid_neighborhood,
                                  grid_node_col, mc_array,
                                  force_prepare_payload);

        break;
      }

      case 3: {
        distributed_overlap<CoordT, edge_wrap, only_neighborhood, true, true,
                            true>(dpi.target_side_info_, target_side->second,
                                  dist_tns_source, tile_grid, grid_neighborhood,
                                  grid_node_col, mc_array,
                                  force_prepare_payload);

        break;
      }

      default: {
        distributed_overlap<CoordT, edge_wrap, only_neighborhood, true, false,
                            false>(dpi.target_side_info_, target_side->second,
                                   dist_tns_source, tile_grid,
                                   grid_neighborhood, grid_node_col, mc_array,
                                   force_prepare_payload);

        break;
      }
      }
    }
  }

  is_source = !dpi.source_side_info_.empty();
  is_target = !dpi.target_side_info_.empty();

  return std::make_pair(is_source || is_target, is_source && is_target);
}
} // namespace sapi

#endif
