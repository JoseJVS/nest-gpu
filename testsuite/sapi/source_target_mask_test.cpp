/*
 *  source_target_mask_test.cpp
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

// Source and target masks select exactly the source and target nodes inside
// them: pairwise_bernoulli with probability 1 and a circular blueprint without
// offset must connect every node pair that passes the masks and the blueprint,
// on every rank, compared with a brute force over the nodes of all ranks.

#include <set>
#include <string>
#include <tuple>
#include <vector>

#include "api_converters.h"
#include "c_api.h"
#include "mask.h"
#include "test_helpers.h"
#include "vp_interface.h"

using namespace sapi;

GC gc;
CAPI capi;

struct GridDef {
  std::string shape;
  std::vector<tileidx_t> dims;
  std::vector<space_t> sides;
};

struct MaskDef {
  std::string name; // empty: no mask
  MASK_SHAPE shape;
  std::vector<space_t> origin;
  std::vector<space_t> params;
};

struct Node {
  vp_t rank;
  nodeidx_t index;
  Coord2D coord;
};

void fill(CharArray &c, const std::string &s) {
  if (!s.empty())
    copy_to_charray_from_string(c, s, gc);
}

void fill(SpaceTArray &a, const std::vector<space_t> &v) {
  if (!v.empty())
    copy_to_array_from_collection(a, v, gc);
}

std::size_t setup_population(const largenodeidx_t count, const uint8_t mode,
                             std::vector<nodeidx_t> &next_index) {
  const auto counts = capi.generate_nodes_in_grid(count, TileIdxArray{}, mode);
  RankNodeSequenceMap sequences;
  for (vp_t rank = 0; rank < capi.get_num_processes(); ++rank) {
    if (counts[rank] < 1)
      continue;
    sequences.emplace(rank, NodeSequence(next_index[rank], counts[rank]));
    next_index[rank] += counts[rank];
  }
  return capi.generate_nodes_in_tiles(sequences, mode);
}

// Local nodes of a population (coordinates are planar: all x, then all y)
std::vector<Node> local_nodes(const std::size_t population) {
  const auto *view = capi.view_node_positions({true, population}, {});
  const auto n = view->node_count_;
  std::vector<Node> nodes;
  for (std::size_t i = 0; i < n; ++i)
    nodes.push_back(
        {capi.get_rank(), view->indexes_[i],
         construct_coord_2D(view->coordinates_[i], view->coordinates_[i + n])});
  capi.free_view_gc();
  return nodes;
}

std::vector<Node> all_nodes(const std::vector<Node> &local) {
  std::vector<double> send;
  for (const auto &node : local) {
    send.push_back(node.rank);
    send.push_back(node.index);
    send.push_back(node.coord.x_);
    send.push_back(node.coord.y_);
  }
  std::vector<double> recv;
  std::vector<int> counts, displacements;
  if (coordinated_all_gatherv_as_ul(send, recv, counts, displacements) == 0)
    recv = send;

  std::vector<Node> nodes;
  for (std::size_t i = 0; i < recv.size(); i += 4)
    nodes.push_back({static_cast<vp_t>(recv[i]),
                     static_cast<nodeidx_t>(recv[i + 1]),
                     construct_coord_2D(recv[i + 2], recv[i + 3])});
  return nodes;
}

void test_masks(const std::vector<Node> &sources, const std::size_t source_pop,
                const std::vector<Node> &local_targets,
                const std::size_t target_pop, const MaskDef &source_mask,
                const MaskDef &target_mask) {
  const space_t radius = 1.2;
  MPStruct mps;
  fill(mps.mask_blueprint_name_, "circular");
  fill(mps.mask_blueprint_params_, {radius});
  fill(mps.source_mask_name_, source_mask.name);
  fill(mps.source_mask_origin_, source_mask.origin);
  fill(mps.source_mask_params_, source_mask.params);
  fill(mps.target_mask_name_, target_mask.name);
  fill(mps.target_mask_origin_, target_mask.origin);
  fill(mps.target_mask_params_, target_mask.params);

  CPStruct cps;
  cps.allow_self_connections_ = true;
  fill(cps.rule_, "pairwise_bernoulli");
  fill(cps.prob_df_name_, "constant");
  fill(cps.prob_df_params_, {1.});
  fill(cps.weight_df_name_, "constant");
  fill(cps.weight_df_params_, {1.});
  fill(cps.delay_df_name_, "constant");
  fill(cps.delay_df_params_, {1.});

  const auto [conn_index, dci] =
      capi.compute_spatial_connections(source_pop, target_pop, mps, cps);
  (void)dci;

  // Incoming connections: source rank, source index, local target index
  std::set<std::tuple<vp_t, nodeidx_t, nodeidx_t>> connections;
  const auto &incoming = capi.view_spatial_connections(conn_index)->first_;
  for (std::size_t r = 0; r < incoming.size_; ++r) {
    const auto &view = incoming.array_[r].second_;
    for (std::size_t p = 0; p < view.num_partitions_; ++p)
      for (std::size_t c = 0; c < view.partition_sizes_[p]; ++c)
        assert(connections
                   .emplace(incoming.array_[r].first_,
                            static_cast<nodeidx_t>(view.sources_[p][c]),
                            static_cast<nodeidx_t>(view.targets_[p][c]))
                   .second);
  }
  capi.clear_spatial_connections(conn_index);
  capi.free_view_gc();

  const auto blueprint =
      construct_mask<Coord2D>(MASK_SHAPE::CIRCULAR, {}, {radius}, {});
  const auto in = [](const MaskDef &def, const Coord2D &c) {
    return def.name.empty() ||
           construct_mask<Coord2D>(def.shape, def.origin, def.params, {})
               .coord_in_mask(c)
               .first;
  };

  std::size_t expected = 0;
  for (const auto &t : local_targets) {
    if (!in(target_mask, t.coord))
      continue;
    for (const auto &s : sources)
      if (in(source_mask, s.coord) &&
          blueprint.coord_in_mask(s.coord, t.coord).first) {
        assert(connections.count({s.rank, s.index, t.index}));
        ++expected;
      }
  }
  assert(connections.size() == expected);
}

int main(int argc, char **argv) {
  init_mpi(&argc, &argv);

  capi.set_rank(get_mpi_rank());
  capi.set_num_processes(get_num_mpi_processes());
  capi.set_num_threads(2);

  const std::vector<GridDef> grids{{"rectangle", {4, 3}, {1.}},
                                   {"rectangle", {3, 4}, {2., 1.}},
                                   {"triangle", {4, 4}, {1.}},
                                   {"hexagon", {2, 4}, {1.}}};
  const std::vector<MaskDef> masks{
      {"circular", MASK_SHAPE::CIRCULAR, {0.5, -0.3}, {0.35}},
      {"circular", MASK_SHAPE::CIRCULAR, {1.3, 1.1}, {0.9}},
      {"elliptical", MASK_SHAPE::ELLIPTICAL, {2., 2.}, {2., 0.3, 45.}},
      {"parallelogram",
       MASK_SHAPE::PARALLELOGRAM,
       {2.5, 1.5},
       {1., 0.2, -0.3, 0.8}},
      // A small ring inside a tile, a wide ring with a hole, and a disc
      {"annular", MASK_SHAPE::ANNULAR, {1.3, 1.1}, {0.2, 0.4}},
      {"annular", MASK_SHAPE::ANNULAR, {2., 2.}, {1., 1.8}},
      {"annular", MASK_SHAPE::ANNULAR, {0.5, 1.}, {0., 0.9}}};
  const MaskDef none{"", MASK_SHAPE::NULL_MS, {}, {}};

  for (const auto &g : grids) {
    capi.reset();
    gc.free_gc();
    capi.set_rng_seed(4242);

    tileidx_t num_tiles = 1;
    for (const auto d : g.dims)
      num_tiles *= d;
    std::vector<std::set<tileidx_t>> ownership(capi.get_num_processes());
    for (tileidx_t t = 0; t < num_tiles; ++t)
      ownership[t % capi.get_num_processes()].insert(t);

    GPStruct gps;
    copy_to_array_from_collection(gps.grid_origin_, std::vector<space_t>{0, 0},
                                  gc);
    copy_to_array_from_collection(gps.grid_dimensions_, g.dims, gc);
    copy_to_charray_from_string(gps.tile_type_, g.shape, gc);
    copy_to_array_from_collection(gps.tile_side_lengths_, g.sides, gc);
    gps.compute_splits_ = false;
    gps.num_splits_ = 2;
    capi.generate_tile_grid(*nested_collection_to_nested_array(ownership, gc),
                            gps);

    std::vector<nodeidx_t> next_index(capi.get_num_processes(), 0);
    const auto source_pop = setup_population(300, 0, next_index);
    const auto target_pop = setup_population(300, 2, next_index);
    const auto sources = all_nodes(local_nodes(source_pop));
    const auto targets = local_nodes(target_pop);

    for (std::size_t i = 0; i < masks.size(); ++i) {
      test_masks(sources, source_pop, targets, target_pop, masks[i], none);
      test_masks(sources, source_pop, targets, target_pop, none, masks[i]);
      test_masks(sources, source_pop, targets, target_pop, masks[i],
                 masks[(i + 2) % masks.size()]);
    }
  }

  capi.reset();
  gc.free_gc();
  finalize_mpi();
  return 0;
}
