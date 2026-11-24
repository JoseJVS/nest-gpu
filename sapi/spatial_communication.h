/*
 *  spatial_communication.h
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

#ifndef SPATIAL_COMMUNICATION_H
#define SPATIAL_COMMUNICATION_H

#include <cassert>
#include <unordered_set>

#include "dispatch.h"
#include "mask_node_processing.h"
#include "mask_tile_processing.h"
#include "sapi_forward_declarations.h"
#include "timer_manager.h"
#include "vp_interface.h"

namespace sapi {
// Forward definition to link with payload_preparation.h
template <Coordinate CoordT>
void reconstruct_received_info(CommunicationInfo<CoordT> &comm_info,
                               RemoteIndexedCoordCache<CoordT> &remote_cache);

template <Coordinate CoordT, bool inverted_source_target>
void compute_target_connections(RankConnectionInfo &rci,
                                RankPairInfo<CoordT> &rpi,
                                const TAArray<MaskCollection<CoordT>> &mc_array,
                                const TAArray<ConnectionGenerator> &cg_array,
                                const RandomManager &rng_manager,
                                const TimerManager &timer_manager,
                                const vp_t target_rank) {
  const auto thread_timer_registry = timer_manager.get_thread_registry();

  ScopedTimer node_disp_timer(thread_timer_registry,
                              "compute_node_displacements_time");

  compute_displacement_checks_across_tile_pairs<CoordT, inverted_source_target>(
      rci, rpi, mc_array, cg_array, rng_manager, target_rank);

  node_disp_timer.stop();

  ScopedTimer map_consolidation_timer(thread_timer_registry,
                                      "consolidate_connections_time");

  rci.consolidate_connection_map();
}

// Checks the arguments of compute_distributed_spatial_connections without
// drawing random numbers. Uses only data every rank holds, so all ranks throw
// the same std::invalid_argument.
template <Coordinate CoordT>
void validate_spatial_connections(
    const DistributedTiledNodeSequenceMap &dist_tns_source,
    const DistributedTiledNodeSequenceMap &dist_tns_target,
    const TileGrid<CoordT> &tile_grid,
    const GridNeighborhood &grid_neighborhood,
    const TAArray<MaskCollection<CoordT>> &mc_array) {
  if (!mc_array.get_local_thread_item()->has_blueprint())
    throw std::invalid_argument("Spatial connections require mask blueprints");

  // Largest payload any rank can send (all its owned tiles, leaves and nodes
  // of either population) has to fit one MPI message; every rank knows every
  // rank's counts, so all ranks reach the same decision
  if (0 < tile_grid.num_tiles_) {
    const int64_t leaves_per_tile = compute_leaves_count<CoordT>(
        tile_grid.splits_, tile_grid.positions_[0].tile_.shape_);
    for (vp_t rank = 0; rank < grid_neighborhood.num_processes_; ++rank) {
      const int64_t tiles = static_cast<int64_t>(
          grid_neighborhood
              .rank_tiles_ownership_map_[static_cast<std::size_t>(rank)]
              .size());
      if (!payload_sizes(tiles, tiles * leaves_per_tile,
                         std::max(count_rank_nodes(dist_tns_source, rank),
                                  count_rank_nodes(dist_tns_target, rank)),
                         CoordT::D)
               .fits_message())
        throw std::invalid_argument(
            "Too many nodes on rank " + std::to_string(rank) +
            " to exchange them in one connection payload");
    }
  }
}

template <Coordinate CoordT>
DistributedConnectionInfo compute_distributed_spatial_connections(
    const DistributedTiledNodeSequenceMap &dist_tns_source,
    const DistributedTiledNodeSequenceMap &dist_tns_target,
    const TileGrid<CoordT> &tile_grid,
    const GridNeighborhood &grid_neighborhood,
    const GridNodeCollection<CoordT> &grid_node_col,
    const TAArray<MaskCollection<CoordT>> &mc_array,
    const TAArray<ConnectionGenerator> &cg_array,
    const RandomManager &rng_manager, const TimerManager &timer_manager,
    const bool edge_wrap, const bool only_neighborhood) {
  assert(timer_manager.is_initialized());

  validate_spatial_connections(dist_tns_source, dist_tns_target, tile_grid,
                               grid_neighborhood, mc_array);

  const auto rank_timer_registry = timer_manager.get_rank_registry();

  ScopedTimer connect_timer(rank_timer_registry,
                            "compute_distributed_spatial_connections_time");

  DistributedConnectionInfo dci;
  DistributedPairInfo<CoordT> dpi;
  RankPairInfo<CoordT> local_pairings;

#pragma omp parallel default(none)                                             \
    shared(dci, dpi, local_pairings, dist_tns_source, dist_tns_target,         \
               tile_grid, grid_neighborhood, grid_node_col, mc_array,          \
               cg_array, rng_manager, timer_manager)                           \
    firstprivate(rank_timer_registry, edge_wrap, only_neighborhood)
#pragma omp master
#pragma omp taskgroup
  {
    ScopedTimer tile_overlap_timer(rank_timer_registry,
                                   "compute_distributed_tile_overlap_time");

    const std::pair<bool, bool> available_targets = dispatch_bools(
        [&]<bool edge_wrap_v, bool only_neighborhood_v>() {
          return compute_distributed_tile_overlap<CoordT, edge_wrap_v,
                                                  only_neighborhood_v>(
              dpi, dist_tns_source, dist_tns_target, tile_grid,
              grid_neighborhood, grid_node_col, mc_array);
        },
        edge_wrap, only_neighborhood);

    tile_overlap_timer.stop();

    ScopedTimer node_overlap_timer(rank_timer_registry,
                                   "dispatch_distributed_node_overlap_time");

    if (available_targets.first) {
      const auto ssi_it =
          dpi.source_side_info_.find(grid_neighborhood.local_rank_);
      const auto tsi_it =
          dpi.target_side_info_.find(grid_neighborhood.local_rank_);
      [[maybe_unused]] const bool source_side =
          ssi_it != dpi.source_side_info_.end();
      [[maybe_unused]] const bool target_side =
          tsi_it != dpi.target_side_info_.end();
      assert(source_side == target_side &&
             source_side == available_targets.second);

      if (available_targets.second) {
        assert(ssi_it->second.sender_info_.data_payload_.empty() &&
               tsi_it->second.sender_info_.data_payload_.empty());

        local_pairings = std::move(ssi_it->second);
        local_pairings.receiver_info_ = std::move(tsi_it->second.sender_info_);

        dpi.source_side_info_.erase(ssi_it);
        dpi.target_side_info_.erase(tsi_it);

        const auto [emplace_it, success] = dci.incoming_connections_.emplace(
            grid_neighborhood.local_rank_, RankConnectionInfo());
        assert(success);

#pragma omp task default(none)                                                 \
    shared(local_pairings, mc_array, cg_array, rng_manager, timer_manager)     \
    firstprivate(emplace_it)
        compute_target_connections<CoordT, false>(
            emplace_it->second, local_pairings, mc_array, cg_array, rng_manager,
            timer_manager, emplace_it->first);
      }

      const bool empty_dpi =
          dpi.source_side_info_.empty() && dpi.target_side_info_.empty();
      assert(empty_dpi <= available_targets.first);

      if (!empty_dpi && grid_neighborhood.num_processes_ > 1) {
        ScopedTimer communicate_timer(
            rank_timer_registry, "dispatch_communicate_node_positions_time");

        communicate_connect_distributed_pair_data(dci, dpi, mc_array, cg_array,
                                                  rng_manager, timer_manager);
      }
    }
  }

  connect_timer.stop();

  ScopedTimer reseed_timer(rank_timer_registry, "seed_update_cleanup_time");

  for (auto &rank_map :
       {&dci.incoming_connections_, &dci.outgoing_connections_}) {
    auto rci_it = rank_map->begin();
    const auto rci_end = rank_map->end();
    while (rci_it != rci_end) {
      rng_manager.update_rank_paired_seed(rci_it->first);

      if (rci_it->second.partitioned_connections_.empty())
        rci_it = rank_map->erase(rci_it);
      else
        ++rci_it;
    }
  }

  return dci;
}

#ifdef HAVE_MPI
struct QueuedRequests {
  std::unordered_set<vp_t> pending_ranks_;
  std::unordered_set<vp_t> done_ranks_;
};

template <Coordinate CoordT>
void communicate_payload_async(
    QueuedRequests &request_queue,
    std::vector<MPI_Request>::iterator &request_it,
    const std::unordered_map<vp_t, RankPairInfo<CoordT>> &rank_info_map,
    const vp_t tag) {
  for (const auto &[rank, rpi] : rank_info_map) {
    assert(!request_queue.pending_ranks_.contains(rank) &&
           rpi.sender_info_.data_payload_.size() <
               std::numeric_limits<vp_t>::max());
    const auto payload_size =
        static_cast<vp_t>(rpi.sender_info_.data_payload_.size());
    assert(0 < payload_size);

    MPI_Isend(rpi.sender_info_.data_payload_.data(), payload_size, MPI_UINT64_T,
              rank, tag, MPI_COMM_WORLD, &(*request_it++));

    request_queue.pending_ranks_.insert(rank);
  }
}

template <Coordinate CoordT>
void probe_payload_recv(
    QueuedRequests &request_queue, MPI_Status &status,
    std::unordered_map<vp_t, RankPairInfo<CoordT>> &rank_info_map,
    const vp_t tag) {
  if (request_queue.pending_ranks_.empty())
    return;

  vp_t flag;
  auto ranks_it = request_queue.pending_ranks_.cbegin();
  const auto ranks_end = request_queue.pending_ranks_.cend();
  while (ranks_it != ranks_end) {
    flag = 0;
    MPI_Iprobe(*ranks_it, tag, MPI_COMM_WORLD, &flag, &status);
    if (flag) {
      MPI_Get_count(&status, MPI_UINT64_T, &flag);
      assert(0 < flag);

      const auto rank_map_it = rank_info_map.find(*ranks_it);
      assert(rank_map_it != rank_info_map.end() &&
             rank_map_it->second.receiver_info_.data_payload_.empty());
      rank_map_it->second.receiver_info_.data_payload_.resize(
          static_cast<size_t>(flag));

      MPI_Recv(rank_map_it->second.receiver_info_.data_payload_.data(), flag,
               MPI_UINT64_T, *ranks_it, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      assert(!request_queue.done_ranks_.contains(*ranks_it));
      request_queue.done_ranks_.insert(*ranks_it);
      ranks_it = request_queue.pending_ranks_.erase(ranks_it);
    } else
      ++ranks_it;
  }
}

template <Coordinate CoordT, bool inverted_source_target>
void dispatch_connection_procedures(
    QueuedRequests &request_queue,
    std::unordered_map<vp_t, RankConnectionInfo> &connection_info_map,
    std::unordered_map<vp_t, RankPairInfo<CoordT>> &rank_info_map,
    const TAArray<MaskCollection<CoordT>> &mc_array,
    const TAArray<ConnectionGenerator> &cg_array,
    const RandomManager &rng_manager, const TimerManager &timer_manager) {
  if (request_queue.done_ranks_.empty())
    return;

  auto ranks_it = request_queue.done_ranks_.cbegin();
  const auto ranks_end = request_queue.done_ranks_.cend();
  while (ranks_it != ranks_end) {
    const auto rank_info_it = rank_info_map.find(*ranks_it);
    assert(rank_info_it != rank_info_map.end());
    ranks_it = request_queue.done_ranks_.erase(ranks_it);

    const auto [emplace_it, success] =
        connection_info_map.emplace(rank_info_it->first, RankConnectionInfo());
    assert(success);

#pragma omp task default(none)                                                 \
    shared(mc_array, cg_array, rng_manager, timer_manager)                     \
    firstprivate(emplace_it, rank_info_it)
    {
      reconstruct_received_info(
          rank_info_it->second.receiver_info_,
          rank_info_it->second.remote_indexed_coord_cache_);
      compute_target_connections<CoordT, inverted_source_target>(
          emplace_it->second, rank_info_it->second, mc_array, cg_array,
          rng_manager, timer_manager, rank_info_it->first);
    }
  }
}
#endif

#ifdef HAVE_MPI
// One side of the exchange loop: receives the payloads that arrived (with the
// mirrored tag: source side payloads are sent with tag 0 and target side
// payloads with tag 1) and dispatches the connection computations of the
// ranks done. Returns whether it dispatched.
template <Coordinate CoordT, bool inverted_source_target>
bool probe_and_dispatch(
    QueuedRequests &request_queue, MPI_Status &status,
    std::unordered_map<vp_t, RankConnectionInfo> &connection_info_map,
    std::unordered_map<vp_t, RankPairInfo<CoordT>> &rank_info_map,
    const TAArray<MaskCollection<CoordT>> &mc_array,
    const TAArray<ConnectionGenerator> &cg_array,
    const RandomManager &rng_manager, const TimerManager &timer_manager) {
  probe_payload_recv(request_queue, status, rank_info_map,
                     inverted_source_target ? 0 : 1);

  if (request_queue.done_ranks_.empty())
    return false;

  dispatch_connection_procedures<CoordT, inverted_source_target>(
      request_queue, connection_info_map, rank_info_map, mc_array, cg_array,
      rng_manager, timer_manager);
  return true;
}
#endif

template <Coordinate CoordT>
void communicate_connect_distributed_pair_data(
    [[maybe_unused]] DistributedConnectionInfo &distributed_connection_map,
    [[maybe_unused]] DistributedPairInfo<CoordT> &distributed_pair_data,
    [[maybe_unused]] const TAArray<MaskCollection<CoordT>> &mc_array,
    [[maybe_unused]] const TAArray<ConnectionGenerator> &cg_array,
    [[maybe_unused]] const RandomManager &rng_manager,
    [[maybe_unused]] const TimerManager &timer_manager) {
#ifdef HAVE_MPI
  const auto num_requests = distributed_pair_data.source_side_info_.size() +
                            distributed_pair_data.target_side_info_.size();
  assert(0 < num_requests);

  std::vector<MPI_Request> send_requests(num_requests);
  auto req_it = send_requests.begin();

  QueuedRequests source_side_queue;
  QueuedRequests target_side_queue;

  const auto rank_timer_registry = timer_manager.get_rank_registry();

  ScopedTimer send_async_timer(rank_timer_registry,
                               "send_async_positions_time");

  if (!distributed_pair_data.source_side_info_.empty())
    communicate_payload_async(source_side_queue, req_it,
                              distributed_pair_data.source_side_info_,
                              0 // tag 0 for sending source side positions
    );

  if (!distributed_pair_data.target_side_info_.empty())
    communicate_payload_async(target_side_queue, req_it,
                              distributed_pair_data.target_side_info_,
                              1 // tag 1 for sending target side positions
    );

  assert(req_it == send_requests.end());

  send_async_timer.stop();

  ScopedTimer trigger_send_timer(rank_timer_registry, "trigger_send_timer");

  vp_t flag;
  MPI_Testall(static_cast<int>(send_requests.size()), send_requests.data(),
              &flag, MPI_STATUSES_IGNORE);

  trigger_send_timer.stop();

  ScopedTimer last_dispatch_timer(rank_timer_registry,
                                  "last_compute_connection_dispatch_timer");
  // Stopped by the first dispatch (later stops do nothing): num_requests > 0,
  // so at least one dispatch runs
  ScopedTimer first_dispatch_timer(rank_timer_registry,
                                   "first_compute_connection_dispatch_timer");

  MPI_Status status;
  while (!(source_side_queue.done_ranks_.empty() &&
           source_side_queue.pending_ranks_.empty()) ||
         !(target_side_queue.done_ranks_.empty() &&
           target_side_queue.pending_ranks_.empty())) {
    if (probe_and_dispatch<CoordT, false>(
            source_side_queue, status,
            distributed_connection_map.outgoing_connections_,
            distributed_pair_data.source_side_info_, mc_array, cg_array,
            rng_manager, timer_manager))
      first_dispatch_timer.stop();

    if (probe_and_dispatch<CoordT, true>(
            target_side_queue, status,
            distributed_connection_map.incoming_connections_,
            distributed_pair_data.target_side_info_, mc_array, cg_array,
            rng_manager, timer_manager))
      first_dispatch_timer.stop();
  }

  last_dispatch_timer.stop();

  ScopedTimer wait_send_timer(rank_timer_registry, "wait_send_timer");

  MPI_Waitall(static_cast<int>(send_requests.size()), send_requests.data(),
              MPI_STATUSES_IGNORE);

  // At this point it is safe to clear sender payloads
  for (const auto &rpi_map_ptr : {&distributed_pair_data.source_side_info_,
                                  &distributed_pair_data.target_side_info_})
    for (auto &rpi_pair : *rpi_map_ptr)
      rpi_pair.second.sender_info_.data_payload_.clear();
#endif
}
} // namespace sapi

#endif
