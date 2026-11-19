/*
 *  tileless_rank_test.cpp
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

// Every tile is owned by rank 0: the other ranks own no tiles, get no nodes
// and no connections, but take part in every call without errors or hangs.

#include "api_test_helpers.h"

using namespace sapi;
using sapi::test::ApiTest;

namespace {

void tileless_pipeline_test(ApiTest &t, const nodeidx_t num_nodes,
                            const split_t splits) {
  const bool owns_tiles = t.capi.get_rank() == 0;

  t.reset();
  t.expect_no_throw("grid", [&] {
    t.generate_grid(t.ownership(16, true), {4, 4}, {0.5, 0.5}, "rectangle",
                    splits);
  });

  const auto &grid = *t.capi.view_grid_vertices();
  t.expect((grid.num_tiles_ != 0) == owns_tiles, "owned tiles",
           std::to_string(grid.num_tiles_) + " tiles");

  NodeCountVector counts;
  t.expect_no_throw("distribute",
                    [&] { counts = t.distribute_nodes(num_nodes); });
  for (std::size_t rank = 1; rank < counts.size(); ++rank)
    t.expect(counts[rank] == 0, "no nodes for tile-less ranks");

  std::size_t population = 0;
  t.expect_no_throw("generate",
                    [&] { population = t.generate_nodes(num_nodes); });

  const auto &nodes =
      *t.capi.view_node_positions({true, population}, MPStruct{});
  t.expect(nodes.node_count_ ==
               (owns_tiles ? static_cast<std::size_t>(num_nodes) : 0),
           "local nodes", std::to_string(nodes.node_count_));

  // A mask covering the whole 2 x 2 domain with edge wrap and probability 1:
  // every pair except self connections
  uint64_t connections = 0;
  t.expect_no_throw("connect", [&] {
    connections =
        t.connect(population, "pairwise_bernoulli", "constant", 1, 0, M_SQRT2);
  });
  const auto expected = static_cast<uint64_t>(num_nodes) * (num_nodes - 1);
  t.expect(connections == expected, "all pairs connected",
           std::to_string(connections) + " of " + std::to_string(expected));

  t.expect_no_throw("fixed indegree", [&] {
    connections = t.connect(population, "fixed_indegree", "constant", 1, 3);
  });
  t.expect(connections == 3 * static_cast<uint64_t>(num_nodes),
           "fixed indegree count", std::to_string(connections));
}
} // namespace

int main(int argc, char **argv) {
  init_mpi(&argc, &argv);
  int result = 0;
  {
    ApiTest t;
    for (const split_t splits : {0, 2})
      tileless_pipeline_test(t, 200, splits);
    t.reset();
    result = t.result();
  }
  finalize_mpi();
  return result;
}
