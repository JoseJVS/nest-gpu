/*
 *  api_test_helpers.h
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

// Helpers for tests that drive SAPI through CAPI and check its error paths.
// Failures are reported with the rank and counted instead of asserted, so a
// test reports every failing check and fails through its exit code even when
// asserts are disabled.

#ifndef API_TEST_HELPERS_H
#define API_TEST_HELPERS_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <limits>
#include <set>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "api_converters.h"
#include "c_api.h"
#include "vp_interface.h"

#ifdef HAVE_MPI
#include <mpi.h>
#endif

namespace sapi::test {

struct ApiTest {
  GC gc;
  CAPI capi;
  int failures = 0;

  ApiTest() {
    capi.set_rank(get_mpi_rank());
    capi.set_num_processes(get_num_mpi_processes());
    capi.set_num_threads(2);
  }

  void reset() {
    capi.reset();
    gc.free_gc();
  }

  void fail(const char *check, const std::string &detail) {
    std::printf("RANK %i FAIL %s: %s\n", capi.get_rank(), check,
                detail.c_str());
    std::fflush(stdout);
    ++failures;
  }

  void expect(const bool condition, const char *check,
              const std::string &detail = "") {
    if (!condition)
      fail(check, detail);
  }

  // The call has to throw E with a message containing message_part
  template <typename E, typename F>
  void expect_throw(const char *check, F &&call,
                    const std::string_view message_part) {
    try {
      call();
    } catch (const E &e) {
      if (std::string_view(e.what()).find(message_part) ==
          std::string_view::npos)
        fail(check, "message \"" + std::string(e.what()) +
                        "\" does not contain \"" + std::string(message_part) +
                        "\"");
      return;
    } catch (const std::exception &e) {
      fail(check,
           "wrong exception type, message \"" + std::string(e.what()) + "\"");
      return;
    }
    fail(check, "no exception");
  }

  template <typename F> void expect_no_throw(const char *check, F &&call) {
    try {
      call();
    } catch (const std::exception &e) {
      fail(check, "unexpected exception \"" + std::string(e.what()) + "\"");
    }
  }

  // Exit code: failures on any rank fail the test on every rank
  int result() const {
    int total = failures;
#ifdef HAVE_MPI
    MPI_Allreduce(&failures, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif
    return total == 0 ? 0 : 1;
  }

  // Tile t is owned by rank t % num_processes, or every tile by rank 0
  std::vector<std::set<tileidx_t>>
  ownership(const tileidx_t num_tiles, const bool all_on_rank_zero = false) {
    std::vector<std::set<tileidx_t>> map(
        static_cast<std::size_t>(capi.get_num_processes()));
    for (tileidx_t tile = 0; tile < num_tiles; ++tile)
      map[all_on_rank_zero
              ? 0
              : static_cast<std::size_t>(tile % capi.get_num_processes())]
          .insert(tile);
    return map;
  }

  void generate_grid(const std::vector<std::set<tileidx_t>> &ownership_map,
                     const std::vector<tileidx_t> &dimensions,
                     const std::vector<space_t> &side_lengths,
                     const std::string &tile_type = "rectangle",
                     const split_t splits = 1) {
    GPStruct gps;
    copy_to_array_from_collection(
        gps.grid_origin_, std::vector<space_t>(dimensions.size(), 0), gc);
    copy_to_array_from_collection(gps.grid_dimensions_, dimensions, gc);
    copy_to_charray_from_string(gps.tile_type_, tile_type, gc);
    copy_to_array_from_collection(gps.tile_side_lengths_, side_lengths, gc);
    gps.compute_splits_ = false;
    gps.num_splits_ = splits;
    capi.generate_tile_grid(
        *nested_collection_to_nested_array(ownership_map, gc), gps);
  }

  // 4 x 4 grid of unit squares, tiles spread over all ranks
  void generate_valid_grid() { generate_grid(ownership(16), {4, 4}, {1, 1}); }

  NodeCountVector
  distribute_nodes(const largenodeidx_t num_nodes,
                   const std::vector<tileidx_t> &target_tiles = {},
                   const uint8_t mode = 2) {
    TileIdxArray targets{};
    if (!target_tiles.empty())
      copy_to_array_from_collection(targets, target_tiles, gc);
    return capi.generate_nodes_in_grid(num_nodes, targets, mode);
  }

  // Distributes and generates num_nodes nodes (balanced, over all tiles);
  // returns the index of the node population
  std::size_t generate_nodes(const largenodeidx_t num_nodes) {
    const auto counts = distribute_nodes(num_nodes);
    RankNodeSequenceMap sequences;
    nodeidx_t first = 0;
    for (vp_t rank = 0; rank < capi.get_num_processes(); ++rank) {
      const auto count = counts[static_cast<std::size_t>(rank)];
      if (0 < count)
        sequences.emplace(rank, NodeSequence(first, count));
      first += count;
    }
    return capi.generate_nodes_in_tiles(sequences, 2);
  }

  void set_rng_type(const std::string &name) {
    CharArray type{};
    copy_to_charray_from_string(type, name, gc);
    capi.set_rng_type(type);
  }

  std::string rng_type() { return charray_to_string(*capi.get_rng_type()); }

  // Only "constant" takes a parameter
  static std::vector<space_t> df_params(const std::string &df,
                                        const space_t value) {
    return df == "constant" ? std::vector<space_t>{value}
                            : std::vector<space_t>{};
  }

  // Connections of a population to itself through a circular mask; returns
  // the number of incoming connections of all ranks
  uint64_t connect(const std::size_t population, const std::string &rule,
                   const std::string &prob_df, const space_t prob_param,
                   const count_t counts = 0, const space_t mask_radius = 0.75,
                   const std::string &weight_df = "constant") {
    return count_incoming(compute_connections(
        population, rule, prob_df, prob_param, counts, mask_radius, weight_df));
  }

  // As connect, but returns the index of the connections
  std::size_t compute_connections(const std::size_t population,
                                  const std::string &rule,
                                  const std::string &prob_df,
                                  const space_t prob_param,
                                  const count_t counts = 0,
                                  const space_t mask_radius = 0.75,
                                  const std::string &weight_df = "constant") {
    MPStruct mps;
    copy_to_charray_from_string(mps.mask_blueprint_name_, "circular", gc);
    copy_to_array_from_collection(mps.mask_blueprint_params_,
                                  std::vector<space_t>{mask_radius}, gc);

    CPStruct cps;
    cps.edge_wrap_ = true;
    cps.allow_self_connections_ = false;
    cps.allow_multiplicity_ = true;
    cps.connection_counts_ = counts;
    copy_to_charray_from_string(cps.rule_, rule, gc);
    copy_to_charray_from_string(cps.prob_df_name_, prob_df, gc);
    copy_to_array_from_collection(cps.prob_df_params_,
                                  df_params(prob_df, prob_param), gc);
    copy_to_charray_from_string(cps.weight_df_name_, weight_df, gc);
    copy_to_array_from_collection(cps.weight_df_params_,
                                  df_params(weight_df, 1.0), gc);
    copy_to_charray_from_string(cps.delay_df_name_, "constant", gc);
    copy_to_array_from_collection(cps.delay_df_params_,
                                  std::vector<space_t>{1.0}, gc);

    return capi.compute_spatial_connections(population, population, mps, cps)
        .first;
  }

  // Number of incoming connections of all ranks
  uint64_t count_incoming(const std::size_t index) {
    const auto &view = *capi.view_connection_counts(index);
    uint64_t local = 0;
    for (std::size_t rank = 0; rank < view.incoming_ranks_; ++rank)
      local += view.incoming_counts_[rank];
    return sum_across_ranks(local);
  }

  // Smallest weight of the incoming connections of all ranks (infinity
  // without connections)
  space_t min_incoming_weight(const std::size_t index) {
    space_t local = std::numeric_limits<space_t>::infinity();
    const auto &incoming = capi.view_spatial_connections(index)->first_;
    for (std::size_t rix = 0; rix < incoming.size_; ++rix) {
      const auto &connections = incoming.array_[rix].second_;
      for (std::size_t pix = 0; pix < connections.num_partitions_; ++pix)
        for (std::size_t cix = 0; cix < connections.partition_sizes_[pix];
             ++cix)
          local = std::min<space_t>(local, connections.weights_[pix][cix]);
    }
    space_t global = local;
#ifdef HAVE_MPI
    static_assert(std::is_same_v<space_t, double>, "MPI_DOUBLE below");
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
#endif
    return global;
  }

  static uint64_t sum_across_ranks(const uint64_t value) {
    uint64_t total = value;
#ifdef HAVE_MPI
    MPI_Allreduce(&value, &total, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
#endif
    return total;
  }
};
} // namespace sapi::test

#endif // API_TEST_HELPERS_H
