/*
 *  capi_error_handling_test.cpp
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

// Error paths of CAPI: calls before a grid exists, failed grid generation
// leaving the api usable, messages for unknown creator names, and the RNG
// settings that are locked once random numbers were drawn.

#include <stdexcept>

#include "api_test_helpers.h"
#include "cg_creators.h"
#include "gf_creators.h"
#include "mask_creators.h"
#include "nf_creators.h"
#include "rng_creators.h"
#include "spatial_manager.h"

using namespace sapi;
using sapi::test::ApiTest;

namespace {

void calls_before_grid_test(ApiTest &t) {
  t.reset();
  t.expect_throw<std::runtime_error>(
      "distribute before grid", [&] { t.distribute_nodes(100); },
      "Spatial grid not initialized yet");
  t.expect_throw<std::runtime_error>(
      "view grid before grid", [&] { t.capi.view_grid_vertices(); },
      "Spatial grid not initialized yet");
  t.expect_throw<std::runtime_error>(
      "view connections before grid", [&] { t.capi.view_connection_counts(0); },
      "Spatial grid not initialized yet");
}

// A failed generate_tile_grid leaves no manager behind, so a later valid grid
// can still be generated
void failed_grid_generation_test(ApiTest &t) {
  t.reset();
  t.expect_throw<std::invalid_argument>(
      "three side lengths",
      [&] { t.generate_grid(t.ownership(16), {4, 4}, {1, 1, 1}); },
      "Invalid rectangle length vector");
  t.expect_throw<std::runtime_error>(
      "distribute after failed grid", [&] { t.distribute_nodes(100); },
      "Spatial grid not initialized yet");
  t.expect_no_throw("valid grid after failed grid",
                    [&] { t.generate_valid_grid(); });
  t.expect_no_throw("nodes after failed grid", [&] { t.generate_nodes(100); });

  t.reset();
  t.expect_throw<std::invalid_argument>(
      "four dimensions",
      [&] { t.generate_grid(t.ownership(16), {2, 2, 2, 2}, {1, 1}); },
      "Invalid grid dimensions");
  t.expect_no_throw("valid grid after four dimensions",
                    [&] { t.generate_valid_grid(); });

  t.reset();
  t.expect_throw<std::invalid_argument>(
      "unknown tile type",
      [&] { t.generate_grid(t.ownership(16), {4, 4}, {1, 1}, "hexagonn"); },
      "Unknown name \"hexagonn\"; registered names: hexagon, rectangle, "
      "triangle");
  t.expect_no_throw("valid grid after unknown tile type",
                    [&] { t.generate_valid_grid(); });

  t.reset();
  t.expect_no_throw("first grid", [&] { t.generate_valid_grid(); });
  t.expect_throw<std::runtime_error>(
      "second grid without reset", [&] { t.generate_valid_grid(); },
      "Cannot initialize spatial grid more than once");
}

// An RNG type set before the grid is checked when the grid is generated (the
// registered RNG names depend on the configured RNG width, so only the common
// part of the message is checked)
void unknown_rng_type_test(ApiTest &t) {
  t.reset();
  t.expect_no_throw("store unknown rng type", [&] { t.set_rng_type("pgc"); });
  t.expect_throw<std::invalid_argument>(
      "grid with unknown rng type", [&] { t.generate_valid_grid(); },
      "Unknown name \"pgc\"; registered names: fake, ");
  t.expect_no_throw("known rng type", [&] { t.set_rng_type("pcg"); });
  t.expect_no_throw("grid with known rng type",
                    [&] { t.generate_valid_grid(); });
  t.expect_no_throw("nodes with known rng type",
                    [&] { t.generate_nodes(100); });
}

template <typename RT, typename Init>
void registry_message_test(ApiTest &t, const char *kind, Init initialize,
                           const std::string &expected_names) {
  CreatorRegistry<RT> registry;
  initialize(registry);
  t.expect_throw<std::invalid_argument>(
      kind, [&] { (void)registry.get_creator("no_such_name"); },
      "Unknown name \"no_such_name\"; " + expected_names);
}

void registry_messages_test(ApiTest &t) {
  registry_message_test<ConnectionGenerator>(
      t, "connection rules", [](auto &r) { initialize_cg_registry(r); },
      "registered names: fixed_indegree, fixed_outdegree, "
      "pairwise_bernoulli, pairwise_poisson");
  registry_message_test<Mask<Coord2D>>(
      t, "2D masks", [](auto &r) { initialize_mk_registry(r); },
      "registered names: circular, elliptical, parallelogram, triangular");
  registry_message_test<AnyRNG>(
      t, "random generators", [](auto &r) { initialize_rng_registry(r); },
      "registered names: fake, ");
  registry_message_test<CachedTileCreator<Coord2D>>(
      t, "2D tile types", [](auto &r) { initialize_ctc_registry(r); },
      "registered names: hexagon, rectangle, triangle");
  registry_message_test<DisplacementFunctor>(
      t, "displacement functions", [](auto &r) { initialize_df_registry(r); },
      "registered names: constant, displacement_x, displacement_y, "
      "displacement_z, distance, distance_x, distance_y, distance_z");
}

// Seed, RNG type and thread count are locked once random numbers were drawn,
// and a refused change keeps the previous setting
void rng_settings_test(ApiTest &t) {
  t.reset();
  t.expect_no_throw("grid", [&] { t.generate_valid_grid(); });
  t.expect_no_throw("seed before drawing", [&] { t.capi.set_rng_seed(777); });
  t.expect_no_throw("type before drawing", [&] { t.set_rng_type("pcg"); });
  t.expect_no_throw("threads before drawing",
                    [&] { t.capi.set_num_threads(3); });
  t.expect_no_throw("threads back before drawing",
                    [&] { t.capi.set_num_threads(2); });

  t.expect_no_throw("nodes", [&] { t.generate_nodes(1000); });
  t.expect_throw<std::runtime_error>(
      "seed after drawing", [&] { t.capi.set_rng_seed(999); },
      "after random numbers were drawn");
  t.expect_throw<std::runtime_error>(
      "type after drawing", [&] { t.set_rng_type("fake"); },
      "after random numbers were drawn");
  t.expect_throw<std::runtime_error>(
      "threads after drawing", [&] { t.capi.set_num_threads(3); },
      "after random numbers were drawn");
  t.expect(t.capi.get_rng_seed() == 777, "seed kept after refusal");
  t.expect(t.rng_type() == "pcg", "type kept after refusal", t.rng_type());
  t.expect(t.capi.get_num_threads() == 2, "threads kept after refusal");
  t.expect_no_throw("nodes again", [&] { t.generate_nodes(1000); });

  // Rejected calls draw nothing, so the settings stay open: before a grid
  // exists, and when the manager's own checks reject the arguments
  t.reset();
  t.expect_throw<std::runtime_error>(
      "distribute before grid", [&] { t.distribute_nodes(10); },
      "Spatial grid not initialized yet");
  t.expect_no_throw("grid", [&] { t.generate_valid_grid(); });
  t.expect_throw<std::invalid_argument>(
      "negative number of nodes", [&] { t.distribute_nodes(-1); },
      "Cannot distribute a negative number of nodes");
  t.expect_throw<std::invalid_argument>(
      "target tile out of range", [&] { t.distribute_nodes(10, {16}, 1); },
      "Target tiles must be a non-empty set");
  t.expect_throw<std::out_of_range>(
      "connections without populations",
      [&] { t.connect(0, "pairwise_bernoulli", "constant", 1); }, "");
  t.expect_no_throw("seed after rejected calls",
                    [&] { t.capi.set_rng_seed(5); });
  t.expect_no_throw("type after rejected calls",
                    [&] { t.set_rng_type("pcg"); });
  t.expect_no_throw("threads after rejected calls",
                    [&] { t.capi.set_num_threads(2); });

  // Reset unlocks the settings
  t.expect_no_throw("nodes", [&] { t.generate_nodes(100); });
  t.reset();
  t.expect_no_throw("seed after reset", [&] { t.capi.set_rng_seed(1); });
  t.expect_no_throw("threads after reset", [&] { t.capi.set_num_threads(2); });
  t.expect_no_throw("grid after reset", [&] { t.generate_valid_grid(); });
}
} // namespace

int main(int argc, char **argv) {
  init_mpi(&argc, &argv);
  int result = 0;
  {
    ApiTest t;
    calls_before_grid_test(t);
    failed_grid_generation_test(t);
    unknown_rng_type_test(t);
    registry_messages_test(t);
    rng_settings_test(t);
    t.reset();
    result = t.result();
  }
  finalize_mpi();
  return result;
}
