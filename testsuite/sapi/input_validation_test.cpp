/*
 *  input_validation_test.cpp
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

// Inputs that are rejected before any parallel region: every rank throws the
// same catchable exception and the api stays usable afterwards.

#include <limits>
#include <stdexcept>

#include "api_test_helpers.h"
#include "payload_preparation.h"

using namespace sapi;
using sapi::test::ApiTest;

namespace {

void grid_parameters_test(ApiTest &t) {
  const auto own = t.ownership(16);
  const auto rejects = [&](const char *check, const std::vector<space_t> sides,
                           const std::string &shape, const split_t splits,
                           const char *message) {
    t.reset();
    t.expect_throw<std::invalid_argument>(
        check, [&] { t.generate_grid(own, {4, 4}, sides, shape, splits); },
        message);
  };

  rejects("zero side", {0, 1}, "rectangle", 1, "Invalid 2D side length");
  rejects("negative side", {1, -1}, "rectangle", 1, "Invalid 2D side length");
  rejects("negative triangle side", {-1}, "triangle", 1,
          "Invalid 2D side length");
  rejects("negative hexagon side", {-1}, "hexagon", 1,
          "Invalid hexagon side length");
  rejects("two hexagon sides", {1, 1}, "hexagon", 1,
          "Invalid hexagon length vector");
  // Sides whose squares are not almost zero, but whose half diagonal is
  rejects("tiny radius", {2.5e-8, 2.5e-8}, "rectangle", 0, "Invalid 2D radius");
  rejects("rectangle splits", {1, 1}, "rectangle", 16,
          "Too many splits for the tile shape, at most 15");
  rejects("triangle splits", {1}, "triangle", 31,
          "Too many splits for the tile shape, at most 30");
  rejects("hexagon splits", {1}, "hexagon", 30,
          "Too many splits for the tile shape, at most 29");

  t.reset();
  t.expect_throw<std::invalid_argument>(
      "too many tiles", [&] { t.generate_grid(own, {65536, 65536}, {1, 1}); },
      "Too many tiles in the grid");

  t.reset();
  t.expect_no_throw("several splits", [&] {
    t.generate_grid(own, {4, 4}, {1, 1}, "rectangle", 4);
  });
  t.reset();
  t.expect_no_throw("valid grid after rejections",
                    [&] { t.generate_valid_grid(); });
  t.expect_no_throw("nodes after rejections", [&] { t.generate_nodes(100); });
}

void distribution_test(ApiTest &t) {
  // Tile 15 has no owner
  auto sparse = t.ownership(16);
  for (auto &tiles : sparse)
    tiles.erase(15);

  t.reset();
  t.expect_no_throw("sparse grid",
                    [&] { t.generate_grid(sparse, {4, 4}, {1, 1}); });
  t.expect_throw<std::invalid_argument>(
      "all tiles of a sparse grid", [&] { t.distribute_nodes(100); },
      "Nodes can only be distributed over tiles owned by some rank");
  t.expect_throw<std::invalid_argument>(
      "unowned target tile", [&] { t.distribute_nodes(100, {0, 15}, 1); },
      "Nodes can only be distributed over tiles owned by some rank");
  t.expect_throw<std::invalid_argument>(
      "target tile out of range", [&] { t.distribute_nodes(100, {16}, 1); },
      "Target tiles must be a non-empty set of tile indexes of the grid");
  t.expect_throw<std::invalid_argument>(
      "negative target tile", [&] { t.distribute_nodes(100, {-1}, 1); },
      "Target tiles must be a non-empty set of tile indexes of the grid");
  t.expect_throw<std::invalid_argument>(
      "targets with free mode", [&] { t.distribute_nodes(100, {0}, 0); },
      "free distribution mode");
  t.expect_throw<std::invalid_argument>(
      "unknown mode", [&] { t.distribute_nodes(100, {}, 7); },
      "Invalid distribution mode");
  t.expect_throw<std::invalid_argument>(
      "negative number of nodes", [&] { t.distribute_nodes(-1); },
      "Cannot distribute a negative number of nodes");

  NodeCountVector counts;
  t.expect_no_throw("owned target tiles",
                    [&] { counts = t.distribute_nodes(100, {0, 1}, 1); });
  uint64_t total = 0;
  for (const auto count : counts)
    total += static_cast<uint64_t>(count);
  t.expect(total == 100, "owned target tiles count",
           std::to_string(total) + " nodes");
}

// More nodes than nodeidx_t holds need a balanced distribution over a grid
// with one tile per rank, and at most nodeidx_t max nodes per tile
void large_distribution_test(ApiTest &t) {
  const auto num_processes = t.capi.get_num_processes();
  const largenodeidx_t max_per_tile = std::numeric_limits<nodeidx_t>::max();
  const largenodeidx_t too_many = num_processes * max_per_tile + 1;

  t.reset();
  t.expect_no_throw("bijected grid", [&] {
    t.generate_grid(t.ownership(num_processes), {num_processes, 1}, {1, 1});
  });
  t.expect_throw<std::invalid_argument>(
      "too many nodes per tile", [&] { t.distribute_nodes(too_many); },
      "Distribution of a large number of nodes needs more tiles");
  t.expect_throw<std::invalid_argument>(
      "large with squeezed mode",
      [&] { t.distribute_nodes(max_per_tile + 1, {}, 1); },
      "balanced distribution mode");
  t.expect_throw<std::invalid_argument>(
      "large with target tiles",
      [&] { t.distribute_nodes(max_per_tile + 1, {0}, 2); },
      "only be done over all tiles");

  // Twice as many tiles as ranks: never a bijection
  t.reset();
  t.expect_no_throw("grid with more tiles than ranks", [&] {
    t.generate_grid(t.ownership(2 * num_processes), {2 * num_processes, 1},
                    {1, 1});
  });
  t.expect_throw<std::invalid_argument>(
      "large over a non-bijected grid",
      [&] { t.distribute_nodes(max_per_tile + 1); },
      "only be done over a bijected grid");
}

void connection_parameters_test(ApiTest &t) {
  t.reset();
  t.generate_valid_grid();
  const auto population = t.generate_nodes(200);

  t.expect_throw<std::invalid_argument>(
      "displacement_z weight in 2D",
      [&] {
        t.connect(population, "pairwise_bernoulli", "constant", 0.5, 0, 0.75,
                  "displacement_z");
      },
      "displacement_z and distance_z need a 3D grid");
  t.expect_throw<std::invalid_argument>(
      "distance_z probability in 2D",
      [&] { t.connect(population, "pairwise_bernoulli", "distance_z", 1); },
      "displacement_z and distance_z need a 3D grid");
  t.expect_throw<std::invalid_argument>(
      "fixed indegree without counts",
      [&] { t.connect(population, "fixed_indegree", "constant", 1, 0); },
      "Invalid connection parameters");
  t.expect_throw<std::invalid_argument>(
      "unknown rule",
      [&] { t.connect(population, "pairwise_bernouli", "constant", 1); },
      "Unknown name \"pairwise_bernouli\"");

  uint64_t connections = 0;
  t.expect_no_throw("valid connections after rejections", [&] {
    connections = t.connect(population, "pairwise_bernoulli", "constant", 1);
  });
  t.expect(0 < connections, "valid connections after rejections count");
}

// Node sequences from counts: one per rank with nodes, starting at 0
RankNodeSequenceMap sequences_from(const NodeCountVector &counts) {
  RankNodeSequenceMap sequences;
  for (std::size_t rank = 0; rank < counts.size(); ++rank)
    if (0 < counts[rank])
      sequences.emplace(static_cast<vp_t>(rank), NodeSequence(0, counts[rank]));
  return sequences;
}

// The node sequences handed back after a distribution or insertion have to
// match its counts; a rejected call keeps the counts for a corrected retry
void node_sequences_test(ApiTest &t) {
  const auto num_processes = t.capi.get_num_processes();

  t.reset();
  t.generate_valid_grid();
  const auto counts = t.distribute_nodes(100);
  const auto valid = sequences_from(counts);

  auto missing = valid;
  missing.erase(missing.begin());
  t.expect_throw<std::invalid_argument>(
      "missing rank", [&] { t.capi.generate_nodes_in_tiles(missing, 2); },
      "Missing node sequence for rank");

  auto shorter = valid;
  --shorter.begin()->second.second;
  t.expect_throw<std::invalid_argument>(
      "wrong length", [&] { t.capi.generate_nodes_in_tiles(shorter, 2); },
      "Invalid node sequence for rank");

  auto unknown = valid;
  unknown.emplace(num_processes, NodeSequence(0, 1));
  t.expect_throw<std::invalid_argument>(
      "unknown rank", [&] { t.capi.generate_nodes_in_tiles(unknown, 2); },
      "Node sequence given for unknown rank");

  t.expect_no_throw("valid sequences after rejections",
                    [&] { t.capi.generate_nodes_in_tiles(valid, 2); });

  // A rank without nodes gets no sequence (needs a second rank)
  if (1 < num_processes) {
    const auto rank_zero_only = t.distribute_nodes(10, {0}, 1);
    auto extra = sequences_from(rank_zero_only);
    extra.emplace(1, NodeSequence(0, 1));
    t.expect_throw<std::invalid_argument>(
        "rank without nodes", [&] { t.capi.generate_nodes_in_tiles(extra, 2); },
        "which has no nodes");
    t.expect_no_throw("valid sequences after rank without nodes", [&] {
      t.capi.generate_nodes_in_tiles(sequences_from(rank_zero_only), 2);
    });
  }

  // Insertion: the same check, and the cached coordinates stay for a retry
  const std::vector<space_t> positions{0.5, 0.5, 1.5, 1.5, 2.5, 2.5, 3.5, 3.5};
  const PositionViewStruct view{2, positions.size(), positions.data()};
  NodeCountVector inserted;
  t.expect_no_throw("insert positions", [&] {
    inserted = t.capi.insert_positions_in_grid(view).first;
  });
  t.expect_throw<std::invalid_argument>(
      "insertion without sequences",
      [&] { t.capi.insert_positions_in_tiles(RankNodeSequenceMap{}); },
      "Missing node sequence for rank");
  t.expect_no_throw("insertion with valid sequences", [&] {
    t.capi.insert_positions_in_tiles(sequences_from(inserted));
  });

  // Checked before the positions are read
  const PositionViewStruct too_many{
      2,
      2 * (static_cast<std::size_t>(std::numeric_limits<nodeidx_t>::max()) + 1),
      positions.data()};
  t.expect_throw<std::invalid_argument>(
      "too many positions", [&] { t.capi.insert_positions_in_grid(too_many); },
      "Too many positions to insert");
}

// The largest payload of a rank has to fit one MPI message (int count of
// 64-bit words). Only the size computation is checked: reaching the runtime
// rejection needs about 9e8 nodes on one rank.
void payload_limit_test(ApiTest &t) {
  t.expect(payload_sizes(1, 1, 800'000'000, 2).fits_message(),
           "8e8 nodes fit one payload");
  t.expect(!payload_sizes(1, 1, 900'000'000, 2).fits_message(),
           "9e8 nodes do not fit one payload");
  t.expect(!payload_sizes(std::numeric_limits<int32_t>::max(), 0, 0, 2)
                .fits_message(),
           "too many tiles do not fit one payload");
}
} // namespace

int main(int argc, char **argv) {
  init_mpi(&argc, &argv);
  int result = 0;
  {
    ApiTest t;
    grid_parameters_test(t);
    distribution_test(t);
    large_distribution_test(t);
    connection_parameters_test(t);
    node_sequences_test(t);
    payload_limit_test(t);
    t.reset();
    result = t.result();
  }
  finalize_mpi();
  return result;
}
