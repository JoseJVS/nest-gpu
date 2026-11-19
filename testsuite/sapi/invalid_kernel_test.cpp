/*
 *  invalid_kernel_test.cpp
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

// Kernel values that are negative, zero or NaN count as probability 0 for
// every rule: such candidates are never connected and nothing aborts.

#include <limits>

#include "api_test_helpers.h"

using namespace sapi;
using sapi::test::ApiTest;

namespace {

constexpr nodeidx_t num_nodes = 200;

constexpr const char *rules[] = {"pairwise_bernoulli", "pairwise_poisson",
                                 "fixed_indegree", "fixed_outdegree"};

void zero_probability_test(ApiTest &t, const std::size_t population) {
  for (const auto kernel :
       {space_t(-1), space_t(0), std::numeric_limits<space_t>::quiet_NaN()}) {
    for (const auto *rule : rules) {
      const auto check =
          std::string(rule) + " with constant " + std::to_string(kernel);
      uint64_t connections = 1;
      t.expect_no_throw(check.c_str(), [&] {
        connections = t.connect(population, rule, "constant", kernel, 3);
      });
      t.expect(connections == 0, check.c_str(),
               std::to_string(connections) + " connections");
    }
  }
}

// displacement_x is negative for half of the candidates: those are dropped,
// the others still connect. The weight is displacement_x too, so every
// connection carries the kernel value it was selected with, and all of them
// have to be positive.
void mixed_sign_test(ApiTest &t, const std::size_t population) {
  for (const auto *rule : rules) {
    const auto check = std::string(rule) + " with displacement_x";
    std::size_t index = 0;
    t.expect_no_throw(check.c_str(), [&] {
      index = t.compute_connections(population, rule, "displacement_x", 0, 3,
                                    0.75, "displacement_x");
    });
    const auto connections = t.count_incoming(index);
    t.expect(0 < connections, check.c_str(), "no connections");
    const auto min_weight = t.min_incoming_weight(index);
    t.expect(0 < min_weight, check.c_str(),
             "connection with kernel value " + std::to_string(min_weight));

    // Fixed degrees draw only among positive candidates, so the counts stay
    // exact while enough of them exist. Checked with one process only: fixed
    // degrees are currently drawn per rank pair (improvement plan, N14)
    if (std::string_view(rule) == "fixed_indegree" &&
        t.capi.get_num_processes() == 1)
      t.expect(connections == static_cast<uint64_t>(3 * num_nodes),
               check.c_str(), std::to_string(connections) + " connections");
  }
}
} // namespace

int main(int argc, char **argv) {
  init_mpi(&argc, &argv);
  int result = 0;
  {
    ApiTest t;
    t.reset();
    t.generate_valid_grid();
    const auto population = t.generate_nodes(num_nodes);

    zero_probability_test(t, population);
    mixed_sign_test(t, population);

    uint64_t connections = 0;
    t.expect_no_throw("constant kernel after invalid ones", [&] {
      connections = t.connect(population, "pairwise_bernoulli", "constant", 1);
    });
    t.expect(0 < connections, "constant kernel after invalid ones count");

    t.reset();
    result = t.result();
  }
  finalize_mpi();
  return result;
}
