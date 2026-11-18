/*
 *  node_distribution.h
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

#ifndef NODE_DISTRIBUTION
#define NODE_DISTRIBUTION

#include <array>
#include <cassert>
#include <concepts>
#include <random>
#include <type_traits>
#include <utility>

#include "node_containers.h"
#include "random_manager.h"

namespace sapi {
// Forward definition to link with vp_interface.h
vp_t get_thread_num();
vp_t get_max_omp_threads();

// Leaf index from the partition taken at each split level, most significant
// first; the same mixed-radix numbering the tile split assigns to leaves.
inline tileidx_t
collapse_dimensions(const std::vector<split_t> &branch_sequence,
                    const std::vector<split_t> &split_sequence) {
  assert(!branch_sequence.empty() &&
         branch_sequence.size() == split_sequence.size());
  auto bs_it = branch_sequence.begin();
  tileidx_t index = *bs_it++;
  for (auto ss_it = split_sequence.begin() + 1; ss_it != split_sequence.end();
       ++ss_it)
    index = *bs_it++ + *ss_it * index;
  return index;
}

template <typename I>
  requires std::integral<std::remove_cvref_t<I>>
NodeCountVector safe_convert_vector(std::vector<I> &&count_vector) {
  if constexpr (std::is_same_v<std::remove_cvref_t<I>, nodeidx_t>)
    return std::move(count_vector);
  else {
    NodeCountVector output_copy(count_vector.size());
    auto out_it = output_copy.begin();
    for (const auto &count : count_vector) {
      if (std::cmp_greater_equal(count, 0) &&
          std::cmp_less(count, std::numeric_limits<nodeidx_t>::max()))
        *out_it++ = static_cast<nodeidx_t>(count);
      else
        throw std::runtime_error(
            "Could not distribute node counts over available bins");
    }

    count_vector.clear();

    return output_copy;
  }
}

template <typename I, bool balanced>
  requires std::integral<std::remove_cvref_t<I>>
inline I balance_node_counts_by_bins(I &num_nodes, const tileidx_t num_bins) {
  if constexpr (balanced) {
    const auto balancing = std::div(num_nodes, static_cast<I>(num_bins));
    num_nodes = balancing.rem;
    return balancing.quot;
  }

  return 0;
}

template <typename I>
  requires std::integral<std::remove_cvref_t<I>>
inline std::pair<I, I>
batch_node_counts_by_thread_count(const I num_nodes, const vp_t num_threads) {
  // Batch tile index drawing with steps equal to the total number of threads
  I full_batches = 0, partial_batches = 0;

  // Only batch if running in multithreaded environment
  if (1 < num_threads) {
    const auto batch_steps = std::div(num_nodes, static_cast<I>(num_threads));
    full_batches = batch_steps.quot;
    partial_batches = batch_steps.rem;
  }

  if (full_batches < static_cast<I>(BATCHING_THRESHOLD)) {
    full_batches = 0;
    partial_batches = num_nodes;
  }

  assert((full_batches * num_threads + partial_batches) == num_nodes);

  return {full_batches, partial_batches};
}

// Adds a drawn bin to a dense histogram, or records it (sparse histogram)
template <bool record, typename T>
inline void add_to_bin(std::vector<T> &out, const tileidx_t bin) {
  if constexpr (record)
    out.push_back(bin);
  else
    ++out[static_cast<std::size_t>(bin)];
}

// Stream of a thread for its full batches
template <bool global>
inline AnyRNG &thread_rng(const RandomManager &rng_manager, const vp_t tid) {
  if constexpr (global)
    return *rng_manager.get_tid_synced_rng(tid);
  else
    return *rng_manager.get_tid_specific_rng(tid);
}

// Stream of the rank for the partial batches
template <bool global>
inline AnyRNG &rank_rng(const RandomManager &rng_manager) {
  if constexpr (global)
    return *rng_manager.get_rank_synced_rng();
  else
    return *rng_manager.get_rank_specific_rng();
}

// Draws `batches` bins into out (see add_to_bin), in draw order. The counter
// and the distribution are copies, so a task's firstprivate values are never
// modified and every call draws with its own distribution object.
template <bool balanced, bool record, typename T, typename I>
  requires std::integral<std::remove_cvref_t<I>>
void batched_draw(std::vector<T> &out, I batches, AnyRNG &rng,
                  std::uniform_int_distribution<tileidx_t> distribution,
                  const tileidx_t num_bins) {
  if constexpr (balanced) {
    assert(batches < num_bins && 1 < num_bins);
    std::vector<tileidx_t> indexes(num_bins);
    std::iota(indexes.begin(), indexes.end(), 0);
    std::shuffle(indexes.begin(), indexes.end(), rng);

    while (0 < batches) {
      add_to_bin<record>(out, indexes[batches]);
      --batches;
    }
  } else {
    while (0 < batches) {
      add_to_bin<record>(out, distribution(rng));
      --batches;
    }
  }
}

template <std::size_t N>
tileidx_t collapse_dimensions(const std::vector<tileidx_t> &grid_pos,
                              const std::array<tileidx_t, N> &grid_dimensions) {
  assert(grid_pos.size() == N);
  auto gp_it = grid_pos.rbegin();
  tileidx_t index = *gp_it++;
  for (auto gd_it = grid_dimensions.rbegin() + 1;
       gd_it != grid_dimensions.rend(); ++gd_it)
    index = *gp_it++ + *gd_it * index;
  return index;
}

// Dimension overload: one distribution per dimension, the bin is the
// collapsed position. There is no balanced mode for dimensions.
template <bool balanced, bool record, typename T, typename I, typename D,
          typename DimensionBoundsT>
  requires std::integral<std::remove_cvref_t<I>>
void batched_draw(std::vector<T> &out, I batches, AnyRNG &rng,
                  std::vector<std::uniform_int_distribution<D>> distributions,
                  const DimensionBoundsT &dimensions) {
  static_assert(!balanced);
  std::vector<typename DimensionBoundsT::value_type> dimension_pos(
      dimensions.size(), 0);
  while (0 < batches) {
    auto dist_it = distributions.begin();
    for (auto pos_it = dimension_pos.begin(); pos_it != dimension_pos.end();
         ++pos_it)
      *pos_it =
          static_cast<typename DimensionBoundsT::value_type>((*dist_it++)(rng));
    add_to_bin<record>(out, collapse_dimensions(dimension_pos, dimensions));
    --batches;
  }
}

// Histogram of the full batches plus init_val per bin: every thread draws
// full_batches bins with its own stream (batched_draw with the distribution
// and bounds given). With more bins than draws per thread the drawn bins are
// recorded and added up; otherwise each thread fills a dense buffer and one
// taskloop over bin chunks sums the buffers. The integer sums do not depend
// on the merge order.
template <bool global, bool balanced, typename I, typename DistT,
          typename BoundsT>
  requires std::integral<std::remove_cvref_t<I>>
std::vector<I> thread_histograms(const RandomManager &rng_manager,
                                 const vp_t num_threads,
                                 const tileidx_t num_bins, const I full_batches,
                                 const I init_val, const DistT &distribution,
                                 const BoundsT &bounds) {
  assert(0 < num_threads && 0 < num_bins && 0 < full_batches);
  const auto threads = static_cast<std::size_t>(num_threads);
  const auto bins = static_cast<std::size_t>(num_bins);

  if (std::cmp_greater(num_bins, full_batches)) {
    std::vector<std::vector<tileidx_t>> drawn(threads);

#pragma omp taskloop num_tasks(strict : num_threads) default(none)             \
    shared(rng_manager, distribution, bounds, drawn)                           \
    firstprivate(num_threads, full_batches)
    for (vp_t curr_idx = 0; curr_idx < num_threads; ++curr_idx) {
      auto &curr_drawn = drawn[static_cast<std::size_t>(curr_idx)];
      curr_drawn.reserve(static_cast<std::size_t>(full_batches));
      batched_draw<balanced, true>(curr_drawn, full_batches,
                                   thread_rng<global>(rng_manager, curr_idx),
                                   distribution, bounds);
    }

    std::vector<I> count_vec(bins, init_val);
    for (const auto &thread_drawn : drawn)
      for (const auto bin : thread_drawn)
        add_to_bin<false>(count_vec, bin);

    return count_vec;
  }

  std::vector<std::vector<I>> buffers(threads);

#pragma omp taskloop num_tasks(strict : num_threads) default(none)             \
    shared(rng_manager, distribution, bounds, buffers)                         \
    firstprivate(num_threads, bins, full_batches)
  for (vp_t curr_idx = 0; curr_idx < num_threads; ++curr_idx) {
    auto &curr_buff = buffers[static_cast<std::size_t>(curr_idx)];
    curr_buff.resize(bins, 0);
    batched_draw<balanced, false>(curr_buff, full_batches,
                                  thread_rng<global>(rng_manager, curr_idx),
                                  distribution, bounds);
  }

  auto count_vec = std::move(buffers[0]);

#pragma omp taskloop num_tasks(strict : num_threads) default(none)             \
    shared(buffers, count_vec)                                                 \
    firstprivate(num_threads, bins, threads, init_val)
  for (std::size_t bin = 0; bin < bins; ++bin) {
    I sum = init_val;
    for (std::size_t thread = 1; thread < threads; ++thread)
      sum += buffers[thread][bin];
    count_vec[bin] += sum;
  }

  return count_vec;
}

template <typename I, bool balanced, bool global>
  requires std::integral<std::remove_cvref_t<I>>
NodeCountVector
uniform_distribute_node_counts(I num_nodes, const tileidx_t num_bins,
                               const RandomManager &rng_manager) {
  assert(0 < num_bins &&
         0 < num_nodes // RNG manager initialization is assumed already tested
  );

  // Trivial case
  if (num_bins == 1)
    return safe_convert_vector(std::vector<I>{num_nodes});

  const auto bins = static_cast<std::size_t>(num_bins);
  const auto init_val =
      balance_node_counts_by_bins<I, balanced>(num_nodes, num_bins);
  if (num_nodes == 0)
    return safe_convert_vector(std::vector<I>(bins, init_val));

  const auto num_threads = get_max_omp_threads();
  auto [full_batches, partial_batches] =
      batch_node_counts_by_thread_count(num_nodes, num_threads);

  const std::uniform_int_distribution<tileidx_t> tile_idx_dist(0, num_bins - 1);

  std::vector<I> count_vec;
  if (0 < full_batches)
    count_vec = thread_histograms<global, balanced>(
        rng_manager, num_threads, num_bins, full_batches, init_val,
        tile_idx_dist, num_bins);
  else
    count_vec.assign(bins, init_val);

  if (0 < partial_batches)
    batched_draw<balanced, false>(count_vec, partial_batches,
                                  rank_rng<global>(rng_manager), tile_idx_dist,
                                  num_bins);

  return safe_convert_vector(std::move(count_vec));
}

template <typename I, typename DimensionBoundsT, bool global>
  requires std::integral<std::remove_cvref_t<I>>
NodeCountVector
uniform_distribute_node_counts(I num_nodes, const tileidx_t num_bins,
                               const DimensionBoundsT &dimensions,
                               const RandomManager &rng_manager) {
  // Trivial case
  if (num_bins == 1)
    return safe_convert_vector(std::vector<I>{num_nodes});

  const auto bins = static_cast<std::size_t>(num_bins);
  if (num_nodes == 0)
    return safe_convert_vector(std::vector<I>(bins, 0));

  const auto num_threads = get_max_omp_threads();
  auto [full_batches, partial_batches] =
      batch_node_counts_by_thread_count(num_nodes, num_threads);

  using dist_t = uniform_int_type_t<typename DimensionBoundsT::value_type>;
  std::vector<std::uniform_int_distribution<dist_t>> distributions;
  distributions.reserve(dimensions.size());
  for (const auto &dim : dimensions)
    distributions.emplace_back(0, static_cast<dist_t>(dim - 1));

  std::vector<I> count_vec;
  if (0 < full_batches)
    count_vec = thread_histograms<global, false>(rng_manager, num_threads,
                                                 num_bins, full_batches, I{0},
                                                 distributions, dimensions);
  else
    count_vec.assign(bins, 0);

  if (0 < partial_batches)
    batched_draw<false, false>(count_vec, partial_batches,
                               rank_rng<global>(rng_manager), distributions,
                               dimensions);

  return safe_convert_vector(std::move(count_vec));
}

// Here it is assumed that node sequences in each rank are generated externally
// i.e. this library generates a number of nodes per tiles then aggregates by
// rank then another library instantiates the nodes in the rank and returns the
// sequence of nodes generated in the rank
DistributedTiledNodeSequenceMap consolidate_node_sequences_per_tile_per_rank(
    const RankNodeSequenceMap &node_sequences_per_rank,
    const RankTileIdxNodeCountPairs &node_counts_per_tile_per_rank);
} // namespace sapi

#endif
