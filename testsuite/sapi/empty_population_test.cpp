/*
 *  empty_population_test.cpp
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

// A population of zero nodes (distributing 0 nodes is allowed) is valid
// input everywhere: it has no nodes to view and gives no connections, alone
// or combined with a non-empty population.

#include <tuple>

#include "api_test_helpers.h"

using namespace sapi;
using sapi::test::ApiTest;

namespace {

void empty_population_test(ApiTest &t, const split_t splits) {
  t.reset();
  t.expect_no_throw("grid", [&] {
    t.generate_grid(t.ownership(16), {4, 4}, {0.5, 0.5}, "rectangle", splits);
  });

  std::size_t empty = 0;
  t.expect_no_throw("empty population", [&] { empty = t.generate_nodes(0); });
  std::size_t full = 0;
  t.expect_no_throw("full population", [&] { full = t.generate_nodes(100); });

  // Inserting no positions (an empty view has no pointer) gives an empty
  // population too
  std::size_t inserted = 0;
  t.expect_no_throw("empty insertion", [&] {
    const PositionViewStruct no_positions{2, 0, nullptr};
    const auto [counts, leftovers] =
        t.capi.insert_positions_in_grid(no_positions);
    t.expect(leftovers->coord_count_ == 0, "no leftovers");
    inserted = t.capi.insert_positions_in_tiles(RankNodeSequenceMap{});
  });

  std::size_t viewed = 1;
  t.expect_no_throw("view empty population", [&] {
    viewed = t.capi.view_node_positions({true, empty}, MPStruct{})->node_count_;
  });
  t.expect(viewed == 0, "no nodes in the empty population",
           std::to_string(viewed));

  for (const auto &[source, target, name] :
       {std::tuple{empty, empty, "empty to empty"},
        std::tuple{inserted, full, "inserted empty to full"},
        std::tuple{empty, full, "empty to full"},
        std::tuple{full, empty, "full to empty"}}) {
    uint64_t connections = 1;
    t.expect_no_throw(name, [&] {
      connections = t.count_incoming(
          t.compute_connections(source, "pairwise_bernoulli", "constant", 1, 0,
                                0.75, "constant", target));
    });
    t.expect(connections == 0, name, std::to_string(connections));
  }

  // The api stays usable
  uint64_t connections = 0;
  t.expect_no_throw("full to full", [&] {
    connections = t.connect(full, "pairwise_bernoulli", "constant", 1);
  });
  t.expect(0 < connections, "full to full count");
}
} // namespace

int main(int argc, char **argv) {
  init_mpi(&argc, &argv);
  int result = 0;
  {
    ApiTest t;
    for (const split_t splits : {0, 2})
      empty_population_test(t, splits);
    t.reset();
    result = t.result();
  }
  finalize_mpi();
  return result;
}
