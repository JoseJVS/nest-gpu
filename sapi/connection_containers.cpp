/*
 *  connection_containers.cpp
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

#include <bit>
#include <cassert>
#include <deque>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include "connection_containers.h"
#include "vp_interface.h"

namespace sapi {
void ConnectionVectors::prepare_vectors(const std::size_t size) {
  assert(connection_sources_.empty() && connection_targets_.empty() &&
         connection_weights_.empty() && connection_delays_.empty());

  if (size == 0)
    return;

  connection_sources_.resize(size);
  connection_targets_.resize(size);
  connection_weights_.resize(size);
  connection_delays_.resize(size);

  sizes_ = size;
}

std::unordered_multimap<combined_idx_t, combined_idx_t>
RankConnectionInfo::build_connection_map() const {
  std::unordered_multimap<combined_idx_t, combined_idx_t> conn_map;
  conn_map.reserve(total_generated_connections_);
  const conn_param_t *weights;
  const conn_param_t *delays;
  for (const auto &conn_vec : partitioned_connections_) {
    weights = conn_vec.connection_weights_.data();
    delays = conn_vec.connection_delays_.data();
    for (std::size_t conn_idx = 0; conn_idx < conn_vec.sizes_; ++conn_idx) {
      const conn_index_t weight =
          std::bit_cast<conn_index_t>(weights[conn_idx]);
      const conn_index_t delay = std::bit_cast<conn_index_t>(delays[conn_idx]);

      conn_map.emplace(
          (static_cast<combined_idx_t>(conn_vec.connection_sources_[conn_idx])
           << 32) |
              static_cast<combined_idx_t>(
                  conn_vec.connection_targets_[conn_idx]),
          (static_cast<combined_idx_t>(weight) << 32) |
              static_cast<combined_idx_t>(delay));
    }
  }

  return conn_map;
}

template <bool inverted_pivot>
void copy_to_connection_vector(std::vector<conn_index_t>::iterator &source_it,
                               std::vector<conn_index_t>::iterator &target_it,
                               std::vector<conn_param_t>::iterator &weight_it,
                               std::vector<conn_param_t>::iterator &delay_it,
                               const std::deque<ConnectionInfo> &conn_vec,
                               const conn_index_t pivot,
                               ConnectionBounds &bounds) {
  if constexpr (inverted_pivot) {
    bounds.update_first_last_target(pivot);
    for (const auto &conn : conn_vec) {
      *source_it++ = conn.index_;
      *target_it++ = pivot;
      *weight_it++ = conn.weight_;
      *delay_it++ = conn.delay_;

      bounds.update_first_last_source(conn.index_);
    }
  } else {
    bounds.update_first_last_source(pivot);
    for (const auto &conn : conn_vec) {
      *source_it++ = pivot;
      *target_it++ = conn.index_;
      *weight_it++ = conn.weight_;
      *delay_it++ = conn.delay_;

      bounds.update_first_last_target(conn.index_);
    }
  }
}

template <bool inverted_pivot>
void copy_to_connection_vector(std::vector<conn_index_t>::iterator &source_it,
                               std::vector<conn_index_t>::iterator &target_it,
                               std::vector<conn_param_t>::iterator &weight_it,
                               std::vector<conn_param_t>::iterator &delay_it,
                               const ProceduralConnectivityBlocks &proc_bloc,
                               ConnectionBounds &bounds) {
  for (auto &conn_block : proc_bloc) {
    if (conn_block.second.empty())
      continue;

    copy_to_connection_vector<inverted_pivot>(source_it, target_it, weight_it,
                                              delay_it, conn_block.second,
                                              conn_block.first, bounds);
  }
}

void ConnectionVectors::copy_from_procedural_connections(
    std::vector<ProceduralConnectivityBlocks> &procedural_connections,
    const std::size_t total_procedural_connections, const bool inverted_pivot) {
  prepare_vectors(total_procedural_connections);

  auto source_it = connection_sources_.begin();
  auto target_it = connection_targets_.begin();
  auto weight_it = connection_weights_.begin();
  auto delay_it = connection_delays_.begin();

  if (inverted_pivot) {
    for (auto &proc_bloc : procedural_connections) {
      if (proc_bloc.empty())
        continue;

      copy_to_connection_vector<true>(source_it, target_it, weight_it, delay_it,
                                      proc_bloc, bounds_);
    }
  } else {
    for (auto &proc_bloc : procedural_connections) {
      if (proc_bloc.empty())
        continue;

      copy_to_connection_vector<false>(source_it, target_it, weight_it,
                                       delay_it, proc_bloc, bounds_);
    }
  }

  assert(source_it == connection_sources_.end());
}

// Adds a connection to partition p_index. Partition numbers are produced in
// order (a pivot's or source index's k-th connection goes to partition k), so
// p_index is at most the current number of partitions.
inline void
add_to_partition(std::deque<std::deque<std::pair<conn_index_t, ConnectionInfo>>>
                     &procedural_partitions,
                 const nodeidx_t p_index, const conn_index_t pivot,
                 const ConnectionInfo &conn) {
  assert(0 <= p_index &&
         std::cmp_less_equal(p_index, procedural_partitions.size()));
  if (std::cmp_equal(p_index, procedural_partitions.size()))
    procedural_partitions.emplace_back();

  procedural_partitions[static_cast<std::size_t>(p_index)].emplace_back(pivot,
                                                                        conn);
}

template <bool inverted_pivot>
std::deque<std::deque<std::pair<conn_index_t, ConnectionInfo>>>
generate_connection_partitions(
    std::vector<ProceduralConnectivityBlocks> &procedural_connections) {
  // An outer deque, not a vector: std::deque's move constructor is not
  // noexcept, so a growing vector would copy every partition
  std::deque<std::deque<std::pair<conn_index_t, ConnectionInfo>>>
      procedural_partitions;

  if constexpr (inverted_pivot) {
    std::unordered_map<conn_index_t, nodeidx_t> indexes;
    for (auto &proc_bloc : procedural_connections) {
      for (auto &[pivot, connections] : proc_bloc) {
        for (const auto &conn : connections)
          add_to_partition(procedural_partitions, indexes[conn.index_]++, pivot,
                           conn);

        connections.clear();
      }

      proc_bloc.clear();
    }
  } else {
    for (auto &proc_bloc : procedural_connections) {
      for (auto &[pivot, connections] : proc_bloc) {
        nodeidx_t index = 0;
        for (const auto &conn : connections)
          add_to_partition(procedural_partitions, index++, pivot, conn);

        connections.clear();
      }

      proc_bloc.clear();
    }
  }

  procedural_connections.clear();

  return procedural_partitions;
}

template <bool inverted_pivot>
void consolidate_partitions(
    std::vector<ProceduralConnectivityBlocks> &procedural_connections,
    std::vector<ConnectionVectors> &consolidated_partitions) {
  auto procedural_partitions =
      generate_connection_partitions<inverted_pivot>(procedural_connections);

  const auto num_threads = get_max_omp_threads();
  const auto total_partitions = procedural_partitions.size();
  assert(total_partitions < std::numeric_limits<nodeidx_t>::max());
  consolidated_partitions.resize(total_partitions);

#pragma omp taskloop num_tasks(strict : num_threads) default(none)             \
    shared(procedural_partitions, consolidated_partitions)                     \
    firstprivate(total_partitions)
  for (std::size_t p_index = 0; p_index < total_partitions; ++p_index) {
    auto &procedural_partition = procedural_partitions[p_index];
    auto &partition = consolidated_partitions[p_index];

    assert(!procedural_partition.empty());
    partition.prepare_vectors(procedural_partition.size());

    for (std::size_t c_index = 0; c_index < partition.sizes_; ++c_index) {
      const auto &index_conn_pair = procedural_partition[c_index];

      if constexpr (inverted_pivot) {
        partition.connection_sources_[c_index] = index_conn_pair.second.index_;
        partition.connection_targets_[c_index] = index_conn_pair.first;

        partition.bounds_.update_first_last_source(
            index_conn_pair.second.index_);
        partition.bounds_.update_first_last_target(index_conn_pair.first);
      } else {
        partition.connection_sources_[c_index] = index_conn_pair.first;
        partition.connection_targets_[c_index] = index_conn_pair.second.index_;

        partition.bounds_.update_first_last_source(index_conn_pair.first);
        partition.bounds_.update_first_last_target(
            index_conn_pair.second.index_);
      }

      partition.connection_weights_[c_index] = index_conn_pair.second.weight_;
      partition.connection_delays_[c_index] = index_conn_pair.second.delay_;
    }

    procedural_partition.clear();
  }

  procedural_partitions.clear();
}

void RankConnectionInfo::consolidate_connection_map() {
  assert(partitioned_connections_.empty());

  if (total_generated_connections_ < 1)
    return;

  assert(!procedural_connections_.empty());

  if (partition_connections_) {
    if (sort_by_pool_indexes_)
      consolidate_partitions<true>(procedural_connections_,
                                   partitioned_connections_);
    else
      consolidate_partitions<false>(procedural_connections_,
                                    partitioned_connections_);
  } else {
    partitioned_connections_.resize(1);
    partitioned_connections_[0].copy_from_procedural_connections(
        procedural_connections_, total_generated_connections_,
        sort_by_pool_indexes_);

    procedural_connections_.clear();
  }
}
} // namespace sapi
