/*
 *  node_distribution.cpp
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

#include <cassert>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include "grid_neighborhood.h"
#include "node_distribution.h"

namespace sapi {
DistributedTiledNodeSequenceMap consolidate_node_sequences_per_tile_per_rank(
    const RankNodeSequenceMap &node_sequences_per_rank,
    const RankTileIdxNodeCountPairs &node_counts_per_tile_per_rank) {
  // The sequences have to match the counts of the preceding distribution or
  // insertion: one sequence of exactly that length per rank with nodes, none
  // for the others. Every rank holds every rank's counts, so all ranks reach
  // the same decision before anything is built.
  for (const auto &[rank, node_sequence] : node_sequences_per_rank)
    if (rank < 0 || !std::cmp_less(rank, node_counts_per_tile_per_rank.size()))
      throw std::invalid_argument("Node sequence given for unknown rank " +
                                  std::to_string(rank));

  for (std::size_t rank_index = 0;
       rank_index < node_counts_per_tile_per_rank.size(); ++rank_index) {
    const auto rank = static_cast<vp_t>(rank_index);
    int64_t expected = 0;
    for (const auto &tile_count : node_counts_per_tile_per_rank[rank_index])
      expected += tile_count.second;

    const auto sequence_it = node_sequences_per_rank.find(rank);
    if (sequence_it == node_sequences_per_rank.end()) {
      if (expected != 0)
        throw std::invalid_argument("Missing node sequence for rank " +
                                    std::to_string(rank) + ", which has " +
                                    std::to_string(expected) + " nodes");
      continue;
    }

    if (expected == 0)
      throw std::invalid_argument("Node sequence given for rank " +
                                  std::to_string(rank) +
                                  ", which has no nodes");

    const auto &[first, count] = sequence_it->second;
    if (first < 0 || count != expected)
      throw std::invalid_argument("Invalid node sequence for rank " +
                                  std::to_string(rank) + ": first index " +
                                  std::to_string(first) + ", " +
                                  std::to_string(count) + " nodes, expected " +
                                  std::to_string(expected) + " nodes");
    if (std::numeric_limits<nodeidx_t>::max() - count < first)
      throw std::invalid_argument("Node sequence for rank " +
                                  std::to_string(rank) +
                                  " exceeds the node index range");
  }

  DistributedTiledNodeSequenceMap dist_tns;
  for (const auto &[rank, node_sequence] : node_sequences_per_rank) {
    nodeidx_t first_node_idx = node_sequence.first;
    TileIdxNodeSequenceMap tile_idx_node_sequence_map;
    for (const auto &[tile_index, node_count] :
         node_counts_per_tile_per_rank[static_cast<std::size_t>(rank)]) {
      if (node_count < 1)
        continue;

      const bool emplace_res =
          tile_idx_node_sequence_map
              .emplace(tile_index, NodeSequence(first_node_idx, node_count))
              .second;
      assert(emplace_res);

      first_node_idx += node_count;
    }
    assert((first_node_idx - node_sequence.first) == node_sequence.second);

    // Not empty: a rank with a sequence has nodes (checked above)
    dist_tns[rank].swap(tile_idx_node_sequence_map);
  }

  return dist_tns;
}
} // namespace sapi
