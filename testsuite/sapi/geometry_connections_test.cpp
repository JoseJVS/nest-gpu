/*
 *  geometry_connections_test.cpp
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

// pairwise_bernoulli with probability 1 connects every node pair whose
// displacement, target - (source + offset), is in the blueprint, for offsets
// and blueprint shapes on every grid shape, with and without edge wrap (then
// for some periodic image of the target). Each rank compares its incoming
// connections with a brute force over the nodes of all ranks.

#include <cassert>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include "api_converters.h"
#include "bounding_box.h"
#include "c_api.h"
#include "creator_registry.h"
#include "gf2d_geometry.h"
#include "gf_collection.h"
#include "gf_creators.h"
#include "mask.h"
#include "test_helpers.h"
#include "type_erasure_helpers.h"
#include "vp_interface.h"

using namespace sapi;

GC gc;
CAPI capi;

struct GridDef {
  std::string shape;
  std::vector<tileidx_t> dims;
  std::vector<space_t> sides;
  split_t splits;
};

struct BlueprintDef {
  std::string name;
  MASK_SHAPE shape;
  std::vector<space_t> params;
  std::vector<space_t> offset;
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

// Periodic image shifts, from the grid functors (as the tile images are)
std::vector<Coord2D> image_shifts(const GridDef &g, const bool edge_wrap) {
  std::vector<Coord2D> shifts{construct_coord_2D(0., 0.)};
  if (!edge_wrap)
    return shifts;

  CreatorRegistry<GridTargetPositionShifts<Coord2D>> gscr;
  CreatorRegistry<ShiftedOriginCreator<Coord2D>> socr;
  CreatorRegistry<CachedTileCreator<Coord2D>> ctcr;
  initialize_gsc_registry(gscr);
  initialize_soc_registry(socr);
  initialize_ctc_registry(ctcr);
  const auto gfc = construct_gf_collection<Coord2D>(
      {0., 0.}, g.dims, g.shape, g.sides, {0}, gscr, socr, ctcr);
  const auto origin = gfc.shift_origin({0, 0}, {true, true});
  const auto px =
      gfc.shift_origin({g.dims[0], 0}, {evenTix(g.dims[0]), true}) - origin;
  const auto py =
      gfc.shift_origin({0, g.dims[1]}, {true, evenTix(g.dims[1])}) - origin;
  for (int i = -1; i <= 1; ++i)
    for (int j = -1; j <= 1; ++j)
      if (i != 0 || j != 0)
        shifts.push_back(px * space_t(i) + py * space_t(j));
  return shifts;
}

// 0: outside, 1: inside, 2: borderline (the verdict flips within 1e-9, only
// checked with edge wrap, whose image shifts are computed independently here)
int blueprint_verdict(const Mask<Coord2D> &blueprint, const Coord2D &source,
                      const Coord2D &target,
                      const std::vector<Coord2D> &shifts) {
  const auto in = [&](const Coord2D &perturbation) {
    bool inside = false;
    for (const auto &shift : shifts)
      inside |=
          blueprint.coord_in_mask(source, target + shift + perturbation).first;
    return inside;
  };
  const bool inside = in(construct_coord_2D(0., 0.));
  if (shifts.size() == 1)
    return inside;
  for (const auto &e :
       {construct_coord_2D(1e-9, 0.), construct_coord_2D(-1e-9, 0.),
        construct_coord_2D(0., 1e-9), construct_coord_2D(0., -1e-9)})
    if (in(e) != inside)
      return 2;
  return inside;
}

void test_blueprint(const GridDef &g, const std::vector<Node> &sources,
                    const std::size_t source_pop,
                    const std::vector<Node> &local_targets,
                    const std::size_t target_pop, const BlueprintDef &def,
                    const bool edge_wrap, const bool self_connections) {
  MPStruct mps;
  fill(mps.mask_blueprint_name_, def.name);
  fill(mps.mask_blueprint_params_, def.params);
  fill(mps.mask_blueprint_offset_, def.offset);

  CPStruct cps;
  cps.edge_wrap_ = edge_wrap;
  cps.allow_self_connections_ = self_connections;
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
      construct_mask<Coord2D>(def.shape, {}, def.params, def.offset);
  const auto shifts = image_shifts(g, edge_wrap);

  std::size_t matched = 0;
  for (const auto &t : local_targets)
    for (const auto &s : sources) {
      // Self connections: the same node is the same population, rank and index
      if (!self_connections && source_pop == target_pop && s.rank == t.rank &&
          s.index == t.index)
        continue;
      const bool connected = connections.count({s.rank, s.index, t.index});
      switch (blueprint_verdict(blueprint, s.coord, t.coord, shifts)) {
      case 0:
        assert(!connected);
        break;
      case 1:
        assert(connected);
        ++matched;
        break;
      default:
        matched += connected;
        break;
      }
    }
  assert(connections.size() == matched);
}

int main(int argc, char **argv) {
  init_mpi(&argc, &argv);

  capi.set_rank(get_mpi_rank());
  capi.set_num_processes(get_num_mpi_processes());
  capi.set_num_threads(2);

  const std::vector<GridDef> grids{{"rectangle", {4, 3}, {1.}, 2},
                                   {"rectangle", {3, 4}, {2., 1.}, 2},
                                   {"triangle", {4, 4}, {1.}, 2},
                                   {"hexagon", {2, 4}, {1.}, 1}};
  const std::vector<BlueprintDef> blueprints{
      {"circular", MASK_SHAPE::CIRCULAR, {0.6}, {}},
      {"circular", MASK_SHAPE::CIRCULAR, {0.3}, {}},
      {"circular", MASK_SHAPE::CIRCULAR, {0.3}, {1., 0.}},
      {"circular", MASK_SHAPE::CIRCULAR, {0.3}, {2., 0.}},
      {"circular", MASK_SHAPE::CIRCULAR, {0.3}, {0., 0.5}},
      {"circular", MASK_SHAPE::CIRCULAR, {0.25}, {-0.2, -0.9}},
      {"elliptical", MASK_SHAPE::ELLIPTICAL, {1.5, 0.15, 45.}, {}},
      {"elliptical", MASK_SHAPE::ELLIPTICAL, {2., 0.2, 45.}, {0.5, 0.5}},
      {"parallelogram",
       MASK_SHAPE::PARALLELOGRAM,
       {0.8, 0., 0.2, 0.4},
       {0.5, 0.5}},
      {"triangular", MASK_SHAPE::TRIANGULAR, {0.9, 0., 0., 0.9}, {}}};

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
    gps.num_splits_ = g.splits;
    capi.generate_tile_grid(*nested_collection_to_nested_array(ownership, gc),
                            gps);

    std::vector<nodeidx_t> next_index(capi.get_num_processes(), 0);
    const auto source_pop = setup_population(300, 0, next_index);
    const auto target_pop = setup_population(300, 2, next_index);
    const auto local_sources = local_nodes(source_pop);
    const auto sources = all_nodes(local_sources);
    const auto targets = local_nodes(target_pop);

    for (const bool edge_wrap : {false, true}) {
      for (const auto &def : blueprints)
        test_blueprint(g, sources, source_pop, targets, target_pop, def,
                       edge_wrap, false);
      for (const bool self_connections : {false, true})
        test_blueprint(g, sources, source_pop, local_sources, source_pop,
                       blueprints[0], edge_wrap, self_connections);
    }
  }

  capi.reset();
  gc.free_gc();
  finalize_mpi();
  return 0;
}
