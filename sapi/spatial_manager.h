/*
 *  spatial_manager.h
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

#ifndef SPATIAL_MANAGER_H
#define SPATIAL_MANAGER_H

#include <cassert>
#include <forward_list>
#include <numeric>

#include "cg_creators.h"
#include "connection_rules.h"
#include "coordinate_geometry.h"
#include "gf2d_geometry.h"
#include "gf_creators.h"
#include "grid_generation.h"
#include "mask2d_geometry.h"
#include "mask_collection.h"
#include "mask_creators.h"
#include "nf_creators.h"
#include "node_containers_init.h"
#include "node_creation.h"
#include "node_insertion.h"
#include "node_slicing.h"
#include "payload_preparation.h"
#include "sapi_forward_declarations.h"
#include "spatial_communication.h"
#include "spatial_containers.h"

namespace sapi {
class BaseSpatialManager {
public:
  BaseSpatialManager() = default;
  BaseSpatialManager(const BaseSpatialManager &) = delete;
  BaseSpatialManager(BaseSpatialManager &&) noexcept = default;
  virtual ~BaseSpatialManager() noexcept = default;

  BaseSpatialManager(const vp_t local_rank, const vp_t num_processes);

  BaseSpatialManager &operator=(const BaseSpatialManager &) = delete;
  BaseSpatialManager &operator=(BaseSpatialManager &&) = delete;

  void set_rng_seed(const rng_seed_t seed);

  void set_rng_type(const std::string &name);

  // True once random numbers were drawn; from then on the operations that
  // reset the RNG streams (seed, type, thread count) are refused. Code that
  // configures OpenMP checks this before changing the thread count.
  bool rng_used() const noexcept;

  const DistributedTiledNodeSequenceMap &
  get_distributed_node_sequence(const std::size_t index) const;

  const DistributedConnectionInfo &
  get_connection_map(const std::size_t index) const;

  ConnectionCounts get_connection_counts(const std::size_t index);

  RecordedTimes get_timer_data() const;

  void clear_connection_map(const std::size_t index);

  virtual bool is_initialized() const = 0;

  virtual void update_num_threads() = 0;

  virtual void
  initialize_tile_grid(std::vector<std::set<tileidx_t>> &&rank_tiles_ownership,
                       const GridParameters &grid_parameters) = 0;

  virtual NodeCountVector distribute_nodes_in_grid(
      const largenodeidx_t num_nodes,
      const std::optional<std::set<tileidx_t>> &target_tiles,
      const DISTRIBUTION_MODE distribution_mode) = 0;

  virtual std::pair<NodeCountVector, std::vector<space_t>>
  insert_positions_in_grid(const dim_t dimensions,
                           const std::size_t coord_count,
                           const space_t *const coordinates) = 0;

  virtual std::size_t
  generate_nodes_in_tiles(const RankNodeSequenceMap &node_sequences_per_rank,
                          const DISTRIBUTION_MODE distribution_mode) = 0;

  virtual std::size_t insert_positions_in_tiles(
      const RankNodeSequenceMap &node_sequences_per_rank) = 0;

  virtual std::pair<std::size_t, DistributedConnectionInfo *>
  compute_spatial_connections(
      const std::size_t source_index, const std::size_t target_index,
      const MaskParameters &mask_parameters,
      const ConnectionParameters &connection_parameters) = 0;

  virtual IndexedNodeCoordinates
  get_nodes(const std::optional<std::size_t> &index,
            const MaskParameters &mask_parameters) = 0;

  virtual GridVertexMap get_grid_vertices() = 0;

protected:
  TimerManager timer_manager_;
  RandomManager random_manager_;
  bool rng_used_ = false;
  GridNeighborhood grid_neighborhood_;
  std::deque<DistributedTiledNodeSequenceMap>
      cached_distributed_tiled_node_sequences_;
  std::deque<DistributedConnectionInfo> cached_connection_maps_;

  void _throw_if_rng_used(const char *const operation) const;
};

inline void BaseSpatialManager::set_rng_seed(const rng_seed_t seed) {
  _throw_if_rng_used("set the RNG seed");
  random_manager_.set_rng_seed(seed);
}

inline void BaseSpatialManager::set_rng_type(const std::string &name) {
  _throw_if_rng_used("set the RNG type");
  random_manager_.set_rng_type(name);
}

inline bool BaseSpatialManager::rng_used() const noexcept { return rng_used_; }

inline void
BaseSpatialManager::_throw_if_rng_used(const char *const operation) const {
  if (rng_used_)
    throw std::runtime_error(std::string("Cannot ") + operation +
                             " after random numbers were drawn");
}

inline const DistributedTiledNodeSequenceMap &
BaseSpatialManager::get_distributed_node_sequence(
    const std::size_t index) const {
  return cached_distributed_tiled_node_sequences_.at(index);
}

inline const DistributedConnectionInfo &
BaseSpatialManager::get_connection_map(const std::size_t index) const {
  return cached_connection_maps_.at(index);
}

inline RecordedTimes BaseSpatialManager::get_timer_data() const {
  return timer_manager_.get_times();
}

inline void BaseSpatialManager::clear_connection_map(const std::size_t index) {
  cached_connection_maps_.at(index).clear();
}

template <Coordinate CoordT>
class SpatialManager final : public BaseSpatialManager {
public:
  SpatialManager();
  SpatialManager(const SpatialManager &) = delete;
  SpatialManager(SpatialManager &&) noexcept = default;

  SpatialManager(const vp_t local_rank, const vp_t num_processes);

  SpatialManager &operator=(const SpatialManager &) = delete;
  SpatialManager &operator=(SpatialManager &&) = delete;

  bool is_initialized() const override;

  void update_num_threads() override;

  void
  initialize_tile_grid(std::vector<std::set<tileidx_t>> &&rank_tiles_ownership,
                       const GridParameters &grid_parameters) override;

  NodeCountVector distribute_nodes_in_grid(
      const largenodeidx_t num_nodes,
      const std::optional<std::set<tileidx_t>> &target_tiles,
      const DISTRIBUTION_MODE distribution_mode) override;

  std::pair<NodeCountVector, std::vector<space_t>>
  insert_positions_in_grid(const dim_t dimensions,
                           const std::size_t coord_count,
                           const space_t *const coordinates) override;

  std::size_t
  generate_nodes_in_tiles(const RankNodeSequenceMap &node_sequences_per_rank,
                          const DISTRIBUTION_MODE distribution_mode) override;

  std::size_t insert_positions_in_tiles(
      const RankNodeSequenceMap &node_sequences_per_rank) override;

  std::pair<std::size_t, DistributedConnectionInfo *>
  compute_spatial_connections(
      const std::size_t source_index, const std::size_t target_index,
      const MaskParameters &mask_parameters,
      const ConnectionParameters &connection_parameters) override;

  IndexedNodeCoordinates
  get_nodes(const std::optional<std::size_t> &index,
            const MaskParameters &mask_parameters) override;

  GridVertexMap get_grid_vertices() override;

protected:
  TileGrid<CoordT> tile_grid_;
  GridNodeCollection<CoordT> grid_collection_;
  bool initialized_ = false;
  CreatorRegistry<UnaryFunctor> uf_registry_;
  CreatorRegistry<Mask<CoordT>> mk_registry_;
  CreatorRegistry<DisplacementFunctor> df_registry_;
  CreatorRegistry<ConnectionGenerator> cg_registry_;
  CreatorRegistry<CachedTileCreator<CoordT>> ctc_registry_;
  CreatorRegistry<GridTargetPositionShifts<CoordT>> gsc_registry_;
  CreatorRegistry<ShiftedOriginCreator<CoordT>> soc_registry_;
  TAArray<GFCollection<CoordT>> gc_array_;
  TAArray<MaskCollection<CoordT>> mc_array_;
  TAArray<ConnectionGenerator> cg_array_;
  TimerRegister *rank_timer_registry_;

  // Temporaries for distribution/insertion of nodes in tiles
  // after getting rank node sequences
  std::forward_list<RankTileIdxNodeCountPairs> temp_node_generation_data_;
  std::forward_list<std::pair<RankTileIdxNodeCountPairs, TiledCoordMap<CoordT>>>
      temp_node_insertion_data_;

  void _initialize_registers();

  void _initialize_managers();

  split_t _initialize_grid_parameters(const GridParameters &grid_parameters);

  void _initialize_mask_parameters(const MaskParameters &mask_parameters);

  void _initialize_connection_parameters(
      const ConnectionParameters &connection_parameters);
};

template <Coordinate CoordT>
SpatialManager<CoordT>::SpatialManager() : BaseSpatialManager() {
  _initialize_registers();
  _initialize_managers();
}

template <Coordinate CoordT>
SpatialManager<CoordT>::SpatialManager(const vp_t local_rank,
                                       const vp_t num_processes)
    : BaseSpatialManager(local_rank, num_processes) {
  _initialize_registers();
  _initialize_managers();
}

template <Coordinate CoordT>
inline void SpatialManager<CoordT>::_initialize_registers() {
  initialize_mk_registry(mk_registry_);
  initialize_uf_registry(uf_registry_);
  initialize_df_registry(df_registry_);
  initialize_cg_registry(cg_registry_);
  initialize_gsc_registry(gsc_registry_);
  initialize_soc_registry(soc_registry_);
  initialize_ctc_registry(ctc_registry_);
}

template <Coordinate CoordT>
inline void SpatialManager<CoordT>::_initialize_managers() {
  timer_manager_.initialize();
  random_manager_.initialize();
  rank_timer_registry_ = timer_manager_.get_rank_registry();
}

template <Coordinate CoordT>
inline bool SpatialManager<CoordT>::is_initialized() const {
  // A rank without tiles has an empty grid collection, so emptiness cannot
  // tell whether the grid was initialized
  assert(!initialized_ ||
         (tile_grid_.has_split_ && grid_neighborhood_.has_owners_ &&
          grid_collection_.empty() ==
              grid_neighborhood_.locally_owned_tiles_->empty()));
  return initialized_;
}

template <Coordinate CoordT>
inline void SpatialManager<CoordT>::update_num_threads() {
  _throw_if_rng_used("update the number of threads");
  gc_array_.prepare();
  mc_array_.prepare();
  cg_array_.prepare();
  _initialize_managers();
}

template <Coordinate CoordT>
split_t SpatialManager<CoordT>::_initialize_grid_parameters(
    const GridParameters &grid_parameters) {
  ScopedTimer gpt(rank_timer_registry_, "grid_param_time");

  const auto gc = construct_gf_collection<CoordT>(
      grid_parameters.grid_origin_, grid_parameters.grid_dimensions_,
      grid_parameters.tile_type_, grid_parameters.tile_side_lengths_,
      grid_parameters.tile_angular_offsets_, gsc_registry_, soc_registry_,
      ctc_registry_);

  // Number of tiles, in a wider type than tileidx_t (the dimensions are
  // positive, see construct_gf_collection)
  int64_t num_tiles = 1;
  for (const auto dimension : gc.grid_dimensions_) {
    num_tiles *= dimension;
    if (std::numeric_limits<tileidx_t>::max() < num_tiles)
      throw std::invalid_argument("Too many tiles in the grid");
  }

  const auto num_splits =
      grid_parameters.compute_splits_
          ? compute_minimal_splits(
                grid_parameters.expected_total_nodes_,
                std::accumulate(gc.grid_dimensions_.cbegin(),
                                gc.grid_dimensions_.cend(), 1,
                                std::multiplies<tileidx_t>{}),
                grid_parameters.expected_nodes_per_leaf_, gc.ctc_.shape_)
          : grid_parameters.num_splits_;

  if (max_splits(gc.ctc_.shape_) < num_splits)
    throw std::invalid_argument("Too many splits for the tile shape, at most " +
                                std::to_string(max_splits(gc.ctc_.shape_)));

  gc_array_.clone(gc);

  return num_splits;
}

template <Coordinate CoordT>
void SpatialManager<CoordT>::initialize_tile_grid(
    std::vector<std::set<tileidx_t>> &&rank_tiles_ownership,
    const GridParameters &grid_parameters) {
  if (is_initialized())
    throw std::runtime_error("Cannot generate tile grid more than once");

  const auto num_splits = _initialize_grid_parameters(grid_parameters);

  ScopedTimer ggt(rank_timer_registry_, "grid_generation_time");

  tile_grid_ = generate_tile_grid(gc_array_);

  ggt.stop();

  ScopedTimer got(rank_timer_registry_, "grid_ownership_time");

  grid_neighborhood_.set_tile_ownership(std::move(rank_tiles_ownership),
                                        tile_grid_);

  got.stop();

  ScopedTimer gst(rank_timer_registry_, "grid_splitting_time");

  split_tiles_in_grid(tile_grid_, *grid_neighborhood_.locally_owned_tiles_,
                      num_splits);

  gst.stop();

  ScopedTimer gnt(rank_timer_registry_, "grid_node_collections_time");

  initialize_local_grid_node_collection(
      grid_collection_, *grid_neighborhood_.locally_owned_tiles_, tile_grid_);
  initialized_ = true;

  gnt.stop();

  ScopedTimer ct(rank_timer_registry_, "cleanup_time");

  gc_array_.clear();
}

template <Coordinate CoordT>
NodeCountVector SpatialManager<CoordT>::distribute_nodes_in_grid(
    const largenodeidx_t num_nodes,
    const std::optional<std::set<tileidx_t>> &target_tiles,
    const DISTRIBUTION_MODE distribution_mode) {
  if (!is_initialized())
    throw std::runtime_error("Spatial grid not initialized");

  ScopedTimer ndt(rank_timer_registry_, "node_distribution_time");

  // The arguments are checked where they are used, before any random number
  // is drawn, so the flag is set only once the call succeeded
  auto [node_counts_per_rank, tiled_node_counts_per_rank] =
      distribute_node_counts_in_grid(num_nodes, target_tiles, tile_grid_,
                                     grid_neighborhood_, random_manager_,
                                     distribution_mode);
  rng_used_ = true;

  temp_node_generation_data_.emplace_front(
      std::move(tiled_node_counts_per_rank));

  return node_counts_per_rank;
}

template <Coordinate CoordT>
std::pair<NodeCountVector, std::vector<space_t>>
SpatialManager<CoordT>::insert_positions_in_grid(
    const dim_t dimensions, const std::size_t coord_count,
    const space_t *const coordinates) {
  if (!is_initialized())
    throw std::runtime_error("Spatial grid not initialized");

  if (dimensions != CoordT::D || coord_count % CoordT::D != 0 ||
      coordinates == nullptr)
    throw std::invalid_argument("Invalid positions to insert");

  ScopedTimer nit(rank_timer_registry_, "node_insertion_time");

  std::deque<CoordT> coords;
  auto read_it = coordinates;
  for (std::size_t index = 0; index < coord_count; index += CoordT::D)
    coords.emplace_back(CoordT::copy_from_vec(read_it));

  rng_used_ = true;
  auto [node_counts_per_rank, tiled_node_counts_per_rank, tiled_coord_map] =
      insert_node_positions_in_grid(coords, tile_grid_, grid_neighborhood_,
                                    random_manager_);

  temp_node_insertion_data_.emplace_front(std::make_pair(
      std::move(tiled_node_counts_per_rank), std::move(tiled_coord_map)));

  std::vector<space_t> leftovers(coords.size() * CoordT::D);
  auto lt_it = leftovers.begin();
  for (const auto &coord : coords)
    coord.copy_to_vec(lt_it);
  coords.clear();

  return std::make_pair(std::move(node_counts_per_rank), std::move(leftovers));
}

template <Coordinate CoordT>
std::size_t SpatialManager<CoordT>::generate_nodes_in_tiles(
    const RankNodeSequenceMap &node_sequences_per_rank,
    const DISTRIBUTION_MODE distribution_mode) {
  if (!is_initialized())
    throw std::runtime_error("Spatial grid not initialized");

  if (temp_node_generation_data_.empty())
    throw std::invalid_argument("Incorrect node generation data cache");

  ScopedTimer nct(rank_timer_registry_, "node_consolidation_time");

  const auto index = cached_distributed_tiled_node_sequences_.size();
  const auto &ref = cached_distributed_tiled_node_sequences_.emplace_back(
      consolidate_node_sequences_per_tile_per_rank(
          node_sequences_per_rank, temp_node_generation_data_.front()));

  temp_node_generation_data_.pop_front();

  nct.stop();

  ScopedTimer ngt(rank_timer_registry_, "node_generation_time");

  // Set on every rank, also those without local tiles, so the guarded
  // operations are refused consistently across ranks
  rng_used_ = true;
  if (const auto local_tns_it = ref.find(grid_neighborhood_.local_rank_);
      local_tns_it != ref.end()) {
    distribute_node_counts_in_tiles(grid_collection_, local_tns_it->second,
                                    tile_grid_, random_manager_,
                                    distribution_mode);
  }

  return index;
}

template <Coordinate CoordT>
std::size_t SpatialManager<CoordT>::insert_positions_in_tiles(
    const RankNodeSequenceMap &node_sequences_per_rank) {
  if (!is_initialized())
    throw std::runtime_error("Spatial grid not initialized");

  if (temp_node_insertion_data_.empty())
    throw std::invalid_argument("Incorrect node insertion data cache");

  ScopedTimer nct(rank_timer_registry_, "node_consolidation_time");

  const auto index = cached_distributed_tiled_node_sequences_.size();
  const auto &ref = cached_distributed_tiled_node_sequences_.emplace_back(
      consolidate_node_sequences_per_tile_per_rank(
          node_sequences_per_rank, temp_node_insertion_data_.front().first));

  nct.stop();

  ScopedTimer ngt(rank_timer_registry_, "node_insertion_time");

  if (const auto local_tns_it = ref.find(grid_neighborhood_.local_rank_);
      local_tns_it != ref.end())
    insert_node_positions_in_tiles(temp_node_insertion_data_.front().second,
                                   grid_collection_, local_tns_it->second,
                                   tile_grid_);
  else if (!temp_node_insertion_data_.front().second.empty())
    throw std::runtime_error("Corrupted tiled coords map cache");

  temp_node_insertion_data_.pop_front();

  return index;
}

template <Coordinate CoordT>
void SpatialManager<CoordT>::_initialize_mask_parameters(
    const MaskParameters &mask_parameters) {
  ScopedTimer mpt(rank_timer_registry_, "mask_param_time");

  const auto mask_collection = create_mask_collection<CoordT>(
      mask_parameters.mask_blueprint_name_,
      mask_parameters.mask_blueprint_params_,
      mask_parameters.mask_blueprint_offset_, mask_parameters.source_mask_name_,
      mask_parameters.source_mask_origin_, mask_parameters.source_mask_params_,
      mask_parameters.source_mask_offset_, mask_parameters.target_mask_name_,
      mask_parameters.target_mask_origin_, mask_parameters.target_mask_params_,
      mask_parameters.target_mask_offset_, mk_registry_);

  mc_array_.clone(mask_collection);
}

template <Coordinate CoordT>
void SpatialManager<CoordT>::_initialize_connection_parameters(
    const ConnectionParameters &connection_parameters) {
  ScopedTimer cpt(rank_timer_registry_, "conn_param_time");

  auto cg = cg_registry_.get_creator(connection_parameters.rule_)->create();
  cg.connection_counts_ = connection_parameters.connection_counts_;
  cg.partition_connections_ = connection_parameters.partition_connections_;
  cg.allow_self_connections_ = connection_parameters.allow_self_connections_;
  cg.allow_multiplicity_ = connection_parameters.allow_multiplicity_;

  cg.numeric_functors_.weight_functor_ = construct_numeric_functor(
      connection_parameters.weight_df_name_,
      connection_parameters.weight_df_params_, df_registry_,
      connection_parameters.weight_ufs_names_,
      connection_parameters.weight_ufs_params_, uf_registry_);

  cg.numeric_functors_.delay_functor_ = construct_numeric_functor(
      connection_parameters.delay_df_name_,
      connection_parameters.delay_df_params_, df_registry_,
      connection_parameters.delay_ufs_names_,
      connection_parameters.delay_ufs_params_, uf_registry_);

  cg.numeric_functors_.probability_functor_ = construct_numeric_functor(
      connection_parameters.prob_df_name_,
      connection_parameters.prob_df_params_, df_registry_,
      connection_parameters.prob_ufs_names_,
      connection_parameters.prob_ufs_params_, uf_registry_);

  // Displacements on a 2D grid have no z component
  if constexpr (CoordT::D < 3) {
    const auto uses_z = [](const NumericFunctor &functor) {
      return functor.df_.func_ == DISPLACEMENT_FUNCTION::DISPLACEMENT_Z ||
             functor.df_.func_ == DISPLACEMENT_FUNCTION::DISTANCE_Z;
    };
    const auto &functors = cg.numeric_functors_;
    if (uses_z(functors.weight_functor_) || uses_z(functors.delay_functor_) ||
        uses_z(functors.probability_functor_))
      throw std::invalid_argument(
          "displacement_z and distance_z need a 3D grid");
  }

  if (!cg.check_parameters())
    throw std::invalid_argument("Invalid connection parameters");

  cg_array_.clone(cg);
}

template <Coordinate CoordT>
std::pair<std::size_t, DistributedConnectionInfo *>
SpatialManager<CoordT>::compute_spatial_connections(
    const std::size_t source_index, const std::size_t target_index,
    const MaskParameters &mask_parameters,
    const ConnectionParameters &connection_parameters) {
  if (!is_initialized())
    throw std::runtime_error("Spatial grid not initialized");

  // Unknown population indexes fail before any creator is built
  const auto &dist_tns_source =
      cached_distributed_tiled_node_sequences_.at(source_index);
  const auto &dist_tns_target =
      cached_distributed_tiled_node_sequences_.at(target_index);

  _initialize_mask_parameters(mask_parameters);
  _initialize_connection_parameters(connection_parameters);

  ScopedTimer sct(rank_timer_registry_, "spatial_conn_time");

  // The remaining arguments are checked where they are used, before any
  // random number is drawn, so the flag is set only once the call succeeded
  const auto index = cached_connection_maps_.size();
  const auto connection_map = &cached_connection_maps_.emplace_back(
      compute_distributed_spatial_connections(
          dist_tns_source, dist_tns_target, tile_grid_, grid_neighborhood_,
          grid_collection_, mc_array_, cg_array_, random_manager_,
          timer_manager_, connection_parameters.edge_wrap_,
          connection_parameters.only_neighborhood_));
  rng_used_ = true;

  sct.stop();

  ScopedTimer ct(rank_timer_registry_, "cleanup_time");

  mc_array_.clear();
  cg_array_.clear();

  return std::make_pair(index, connection_map);
}

template <Coordinate CoordT>
IndexedNodeCoordinates
SpatialManager<CoordT>::get_nodes(const std::optional<std::size_t> &index,
                                  const MaskParameters &mask_parameters) {
  if (!is_initialized())
    throw std::runtime_error("Spatial grid not initialized");

  const auto *const dist_tns =
      index.has_value()
          ? &cached_distributed_tiled_node_sequences_.at(index.value())
          : nullptr;

  _initialize_mask_parameters(mask_parameters);

  ScopedTimer sst(rank_timer_registry_, "spatial_slicing_time");

  const auto slice = slice_tiled_node_maps(
      dist_tns, tile_grid_, grid_neighborhood_, grid_collection_, mc_array_);

  sst.stop();

  ScopedTimer sct(rank_timer_registry_, "slice_conversion_time");

  IndexedNodeCoordinates res;

  std::size_t total_size = 0;
  for (const auto &leaf_nodes : slice)
    total_size += leaf_nodes.size();

  if (0 < total_size) {
    res.dimensions_ = CoordT::D;
    res.indexes_.resize(total_size);
    res.coordinates_.resize(CoordT::D * total_size);

    std::size_t index = 0;
    for (const auto &leaf_nodes : slice) {
      for (const auto &coord_view : leaf_nodes) {
        res.indexes_[index] = coord_view->first;
        res.coordinates_[index] = coord_view->second.x_;
        res.coordinates_[index + total_size] = coord_view->second.y_;

        if constexpr (std::is_same_v<CoordT, Coord3D>) {
          res.coordinates_[index + 2 * total_size] = coord_view->second.z_;
        }
        ++index;
      }
    }
  }

  sct.stop();

  ScopedTimer ct(rank_timer_registry_, "cleanup_time");

  mc_array_.clear();

  return res;
}

template <Coordinate CoordT>
GridVertexMap SpatialManager<CoordT>::get_grid_vertices() {
  if (!is_initialized())
    throw std::runtime_error("Spatial grid not initialized");

  ScopedTimer gvt(rank_timer_registry_, "grid_vertex_time");

  GridVertexMap grid_vertices;
  grid_vertices.dimensions_ = CoordT::D;

  const auto &locally_owned_tiles = *grid_neighborhood_.locally_owned_tiles_;
  if (!locally_owned_tiles.empty()) {
    const auto pos_begin = tile_grid_.positions_.cbegin();
    auto owned_pos = pos_begin + *locally_owned_tiles.cbegin();
    grid_vertices.num_tiles_ = locally_owned_tiles.size();
    grid_vertices.leaves_per_tile_ = owned_pos->tile_.leaf_tiles_.size();
    grid_vertices.vertices_per_tile_ = owned_pos->tile_.vertices_.size();
    grid_vertices.vertices_per_leaf_ =
        (*owned_pos->tile_.leaf_tiles_.cbegin())->vertices_.size();

    const auto total_tile_size = grid_vertices.num_tiles_ *
                                 grid_vertices.vertices_per_tile_ *
                                 grid_vertices.dimensions_;
    const auto total_leaf_size =
        grid_vertices.num_tiles_ * grid_vertices.leaves_per_tile_ *
        grid_vertices.vertices_per_leaf_ * grid_vertices.dimensions_;

    if ((0 < total_leaf_size) != (0 < total_tile_size))
      throw std::runtime_error(
          "Error computing vertex vectors during grid vertices export");

    if (0 < total_tile_size) {
      grid_vertices.tile_indexes_.resize(grid_vertices.num_tiles_);
      grid_vertices.tile_vertices_.resize(total_tile_size);
      grid_vertices.leaf_vertices_.resize(total_leaf_size);
      auto ti_it = grid_vertices.tile_indexes_.begin();
      auto tv_it = grid_vertices.tile_vertices_.begin();
      auto lv_it = grid_vertices.leaf_vertices_.begin();

      for (const auto &owned_tile : locally_owned_tiles) {
        (*ti_it++) = owned_tile;
        owned_pos = pos_begin + owned_tile;

        for (const auto &vertex : owned_pos->tile_.vertices_)
          vertex.copy_to_vec(tv_it);

        for (const auto &leaf_ptr : owned_pos->tile_.leaf_tiles_)
          for (const auto &vertex : leaf_ptr->vertices_)
            vertex.copy_to_vec(lv_it);
      }

      if (ti_it != grid_vertices.tile_indexes_.end())
        throw std::runtime_error("Error exporting owned tile indexes");

      if (tv_it != grid_vertices.tile_vertices_.end())
        throw std::runtime_error("Error exporting owned tile vertices");

      if (lv_it != grid_vertices.leaf_vertices_.end())
        throw std::runtime_error("Error exporting owned leaf vertices");
    }
  }

  return grid_vertices;
}
} // namespace sapi

#endif
