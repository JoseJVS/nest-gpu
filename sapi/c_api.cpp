/*
 *  c_api.cpp
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

#include <span>

#include "api_converters.h"
#include "c_api.h"
#include "spatial_manager.h"

namespace sapi {
static bool INIT_OMP_ONCE = true;

CAPI::CAPI() {
  if (INIT_OMP_ONCE) {
    init_omp(1);
    INIT_OMP_ONCE = false;
  }

  generate_param_name_pair_array(param_name_map_, param_name_gc_);
}

ParameterNamesPairArray *CAPI::get_parameter_names() {
  return &param_name_map_;
}

void CAPI::reset() {
  gc_.free_gc();
  view_gc_.free_gc();
  spatial_storage_.free_gc();
  spatial_manager_ = nullptr;
  seed_ = DEFAULT_BASE_SEED_;
  rng_type_ = DEFAULT_RNG_TYPE_;
}

void CAPI::free_gc() { gc_.free_gc(); }

void CAPI::free_view_gc() { view_gc_.free_gc(); }

vp_t CAPI::get_rank() const { return local_rank_; }

void CAPI::set_rank(const vp_t rank) {
  if (spatial_manager_ != nullptr)
    throw std::runtime_error("API is already initialized");
  if (rank < 0)
    throw std::runtime_error("Invalid MPI rank");

  local_rank_ = rank;
}

vp_t CAPI::get_num_processes() const { return num_processes_; }

void CAPI::set_num_processes(const vp_t num_processes) {
  if (spatial_manager_ != nullptr)
    throw std::runtime_error("API is already initialized");
  if (num_processes < 1)
    throw std::runtime_error("Invalid MPI num processes");

  num_processes_ = num_processes;
}

vp_t CAPI::get_num_threads() const { return get_max_omp_threads(); }

void CAPI::set_num_threads(const vp_t num_threads) {
  // Checked before OpenMP changes, so a refusal leaves the per-thread data
  // sized for the current team
  if (spatial_manager_ != nullptr && spatial_manager_->rng_used())
    throw std::runtime_error(
        "Cannot change the number of threads after random numbers were drawn");

  set_max_omp_threads(num_threads);
  if (spatial_manager_ != nullptr)
    spatial_manager_->update_num_threads();
}

rng_seed_t CAPI::get_rng_seed() const { return seed_; }

void CAPI::set_rng_seed(const rng_seed_t seed) {
  // Stored only once the manager accepted it
  if (spatial_manager_ != nullptr)
    spatial_manager_->set_rng_seed(seed);
  seed_ = seed;
}

CharArray *CAPI::get_rng_type() { return string_to_charray(rng_type_, gc_); }

void CAPI::set_rng_type(const CharArray &rng_type) {
  // Stored only once the manager accepted it
  auto name = charray_to_string(rng_type);
  if (spatial_manager_ != nullptr)
    spatial_manager_->set_rng_type(name);
  rng_type_ = std::move(name);
}

void CAPI::generate_tile_grid(const NestedTileIdxArray &rank_tiles_ownership,
                              const GPStruct &grid_parameters) {
  if (spatial_manager_ != nullptr)
    throw std::runtime_error(
        "Cannot initialize spatial grid more than once without resetting api");

  // Configured and initialized locally, so a failure at any point leaves the
  // api without a manager and the original exception propagates unchanged.
  std::unique_ptr<BaseSpatialManager> manager;
  switch (grid_parameters.grid_dimensions_.size_) {
  case 2:
    manager =
        std::make_unique<SpatialManager<Coord2D>>(local_rank_, num_processes_);
    break;

  case 3:
    manager =
        std::make_unique<SpatialManager<Coord3D>>(local_rank_, num_processes_);
    break;

  default:
    throw std::invalid_argument("Invalid grid dimensions");
  }

  if (seed_ != DEFAULT_BASE_SEED_)
    manager->set_rng_seed(seed_);

  if (rng_type_ != DEFAULT_RNG_TYPE_)
    manager->set_rng_type(rng_type_);

  manager->initialize_tile_grid(
      nested_array_to_set_vector(rank_tiles_ownership),
      gpstruct_to_grid_params(grid_parameters));

  const auto manager_ptr = manager.get();
  spatial_storage_.collect(std::move(manager));
  spatial_manager_ = manager_ptr;
}

BaseSpatialManager &CAPI::_manager() const {
  if (spatial_manager_ == nullptr)
    throw std::runtime_error("Spatial grid not initialized yet");
  return *spatial_manager_;
}

NodeCountVector CAPI::generate_nodes_in_grid(const largenodeidx_t num_nodes,
                                             const TileIdxArray &target_tiles,
                                             const uint8_t distribution_mode) {
  auto &manager = _manager();

  std::optional<std::set<tileidx_t>> set;
  if (target_tiles.size_ > 0)
    set.emplace(array_to_set(target_tiles));

  return manager.distribute_nodes_in_grid(
      num_nodes, set, get_distribution_mode(distribution_mode));
}

std::size_t CAPI::generate_nodes_in_tiles(
    const RankNodeSequenceMap &node_sequences_per_rank,
    const uint8_t distribution_mode) {
  return _manager().generate_nodes_in_tiles(
      node_sequences_per_rank, get_distribution_mode(distribution_mode));
}

std::pair<NodeCountVector, PositionViewStruct *>
CAPI::insert_positions_in_grid(const PositionViewStruct &positions) {
  auto &manager = _manager();

  // No positions (an empty population) may come without a pointer
  if (positions.coordinates_ == nullptr && positions.coord_count_ != 0)
    throw std::invalid_argument("Invalid positions to insert");

  auto [node_counts_per_rank, leftovers] = manager.insert_positions_in_grid(
      positions.dimensions_,
      std::span<const space_t>(positions.coordinates_, positions.coord_count_));

  auto uptr = std::make_unique<std::vector<space_t>>(std::move(leftovers));
  const auto ptr = uptr.get();
  gc_.collect(std::move(uptr));

  return std::make_pair(
      std::move(node_counts_per_rank),
      make_view_from_positions(positions.dimensions_, *ptr, gc_));
}

std::size_t CAPI::insert_positions_in_tiles(
    const RankNodeSequenceMap &node_sequences_per_rank) {
  return _manager().insert_positions_in_tiles(node_sequences_per_rank);
}

std::pair<std::size_t, DistributedConnectionInfo *>
CAPI::compute_spatial_connections(const std::size_t source_index,
                                  const std::size_t target_index,
                                  const MPStruct &mask_parameters,
                                  const CPStruct &connection_parameters) {
  return _manager().compute_spatial_connections(
      source_index, target_index, mpstruct_to_mask_params(mask_parameters),
      cpstruct_to_conn_params(connection_parameters));
}

NodesViewStruct *CAPI::view_node_positions(const OptionalIndex &index,
                                           const MPStruct &mask_parameters) {
  auto &manager = _manager();

  std::optional<std::size_t> opt_idx;
  if (index.first_)
    opt_idx.emplace(index.second_);

  auto uptr = std::make_unique<IndexedNodeCoordinates>(
      manager.get_nodes(opt_idx, mpstruct_to_mask_params(mask_parameters)));
  const auto ptr = uptr.get();
  view_gc_.collect(std::move(uptr));

  return make_view_from_indexed_node_coords(*ptr, gc_);
}

RemoteConnectionViewPair *
CAPI::view_spatial_connections(const std::size_t index) {
  return make_view_from_distributed_connection_info(
      _manager().get_connection_map(index), gc_);
}

ConnectionCountsViewStruct *
CAPI::view_connection_counts(const std::size_t index) {
  auto uptr = std::make_unique<ConnectionCounts>(
      _manager().get_connection_counts(index));
  const auto ptr = uptr.get();
  view_gc_.collect(std::move(uptr));

  return make_view_from_connection_counts(*ptr, gc_);
}

GridViewStruct *CAPI::view_grid_vertices() {
  auto uptr = std::make_unique<GridVertexMap>(_manager().get_grid_vertices());
  const auto ptr = uptr.get();
  view_gc_.collect(std::move(uptr));

  return make_view_from_grid_vertex_map(*ptr, gc_);
}

TiledNodeSequencePairArray *
CAPI::get_distributed_node_sequences(const std::size_t index) {
  return dist_tns_map_to_tns_pair_array(
      _manager().get_distributed_node_sequence(index), gc_);
}

RecordedTimesArrayPair *CAPI::get_timer_data() {
  return recorded_times_to_array_pair(_manager().get_timer_data(), gc_);
}

void CAPI::clear_spatial_connections(const std::size_t index) {
  _manager().clear_connection_map(index);
}
} // namespace sapi
