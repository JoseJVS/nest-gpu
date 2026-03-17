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

#include "c_api.h"
#include "api_converters.h"
#include "spatial_manager.h"


namespace sapi
{
static bool INIT_OMP_ONCE = true;


CAPI::CAPI()
{
    if ( INIT_OMP_ONCE )
    {
        init_omp( 1 );
        INIT_OMP_ONCE = false;
    }
}


void CAPI::reset()
{
    gc_.free_gc();
    spatial_storage_.free_gc();
    spatial_manager_ = nullptr;
    seed_ = DEFAULT_BASE_SEED_;
    rng_type_ = DEFAULT_RNG_TYPE_;
}


void CAPI::free_gc()
{
    gc_.free_gc();
}


vp_t CAPI::get_rank() const
{
    return local_rank_;
}


void CAPI::set_rank( const vp_t& rank )
{
    if ( spatial_manager_ != nullptr )
        throw std::runtime_error( "API is already initialized" );
    if ( rank < 0 )
        throw std::runtime_error( "Invalid MPI rank" );

    local_rank_ = rank;
}


vp_t CAPI::get_num_processes() const
{
    return num_processes_;
}


void CAPI::set_num_processes( const vp_t& num_processes )
{
    if ( spatial_manager_ != nullptr )
        throw std::runtime_error( "API is already initialized" );
    if ( num_processes < 1 )
        throw std::runtime_error( "Invalid MPI num processes" );

    num_processes_ = num_processes;
}


vp_t CAPI::get_num_threads() const
{
    return get_max_omp_threads();
}


void CAPI::set_num_threads( const vp_t& num_threads )
{
    set_max_omp_threads( num_threads );
    if ( spatial_manager_ != nullptr )
        spatial_manager_->update_num_threads();
}

    uint32_t CAPI::get_rng_seed() const
{
    return seed_;
}


void CAPI::set_rng_seed( const uint32_t& seed )
{
    seed_ = seed;
    if ( spatial_manager_ != nullptr )
        spatial_manager_->set_rng_seed( seed );
}


CharArray* CAPI::get_rng_type()
{
    return string_to_charray( rng_type_, gc_ );
}


void CAPI::set_rng_type( const CharArray& rng_type )
{
    rng_type_ = charray_to_string( rng_type );
    if ( spatial_manager_ != nullptr )
        spatial_manager_->set_rng_type( rng_type_ );
}


void CAPI::generate_tile_grid(
    const SpaceTArray& grid_origin,
    const TileIdxArray& grid_dimensions,
    const CharArray& tile_type,
    const SpaceTArray& tile_side_lengths,
    const AngleTArray& tile_angular_offsets,
    const NestedTileIdxArray& rank_tiles_ownership_map,
    const split_t& num_splits
)
{
    if ( spatial_manager_ != nullptr )
        throw std::runtime_error(
            "Cannot initialize spatial grid more than once without resetting api"
        );

    switch ( grid_dimensions.size_ )
    {
    case 2:
    {
        auto uptr = std::make_unique< SpatialManager< Coord2D > >( local_rank_, num_processes_ );
        spatial_manager_ = uptr.get();
        spatial_storage_.collect( std::move( uptr ) );
        break;
    }

    case 3:
    {
        auto uptr = std::make_unique< SpatialManager< Coord3D > >( local_rank_, num_processes_ );
        spatial_manager_ = uptr.get();
        spatial_storage_.collect( std::move( uptr ) );
        break;
    }

    default:
        throw std::invalid_argument( "Incorrect grid dimensions" );
        break;
    }

    if ( seed_ != DEFAULT_BASE_SEED_ )
        spatial_manager_->set_rng_seed( seed_ );

    if ( rng_type_ != DEFAULT_RNG_TYPE_ )
        spatial_manager_->set_rng_type( rng_type_ );

    spatial_manager_->initialize_tile_grid(
        array_to_vector( grid_origin ),
        array_to_vector( grid_dimensions ),
        charray_to_string( tile_type ),
        array_to_vector( tile_side_lengths ),
        array_to_vector( tile_angular_offsets ),
        nested_array_to_set_vector( rank_tiles_ownership_map ),
        num_splits
    );
}


NodeCountVector CAPI::generate_nodes_in_grid(
    const largenodeidx_t& num_nodes,
    const TileIdxArray& tile_set,
    const uint8_t& mode_int
)
{
    if ( spatial_manager_ == nullptr )
        throw std::runtime_error( "Spatial grid not initialized yet" );

    std::optional< std::set< tileidx_t > > set;
    if ( tile_set.size_ > 0 )
        set.emplace( array_to_set( tile_set ) );

    return spatial_manager_->distribute_nodes_in_grid(
        num_nodes,
        set,
        get_distribution_mode( mode_int )
    );
}


std::size_t CAPI::generate_nodes_in_tiles(
    const RankNodeSequenceMap& node_sequence_map,
    const uint8_t& mode_int
)
{
    if ( spatial_manager_ == nullptr )
        throw std::runtime_error( "Spatial grid not initialized yet" );

    return spatial_manager_->generate_nodes_in_tiles(
        node_sequence_map,
        get_distribution_mode( mode_int )
    );
}


std::pair< NodeCountVector, NestedSpaceTArray* >
CAPI::insert_positions_in_grid(
    const NestedSpaceTArray& anycoord_array
)
{
    if ( spatial_manager_ == nullptr )
        throw std::runtime_error( "Spatial grid not initialized yet" );

    auto anycoord_vector = nested_array_to_nested_vector( anycoord_array );

    const auto node_counts_per_rank = spatial_manager_->insert_positions_in_grid( anycoord_vector );

    return std::make_pair(
        node_counts_per_rank,
        nested_collection_to_nested_array( anycoord_vector, gc_ )
    );
}


std::size_t CAPI::insert_positions_in_tiles(
    const RankNodeSequenceMap& node_sequence_map
)
{
    if ( spatial_manager_ == nullptr )
        throw std::runtime_error( "Spatial grid not initialized yet" );

    return spatial_manager_->insert_positions_in_tiles(
        node_sequence_map
    );
}


std::pair< std::size_t, RankConnectionInfo* >
CAPI::compute_spatial_connections(
    const std::size_t& dist_tns_source_index,
    const std::size_t& dist_tns_target_index,
    const MPStruct& mask_params,
    const CPStruct& conn_params
)
{
    if ( spatial_manager_ == nullptr )
        throw std::runtime_error( "Spatial grid not initialized yet" );

    return spatial_manager_->compute_spatial_connections(
        dist_tns_source_index,
        dist_tns_target_index,
        mpstruct_to_mask_params( mask_params ),
        cpstruct_to_conn_params( conn_params )
    );
}


NestedNodeCoordPairArray*
CAPI::get_nodes(
    const OptionalIndex& opt_dist_tns_index,
    const MPStruct& mask_params
)
{
    if ( spatial_manager_ == nullptr )
        throw std::runtime_error( "Spatial grid not initialized yet" );

    std::optional< std::size_t > opt_dtns_idx;
    if ( opt_dist_tns_index.first_ )
        opt_dtns_idx.emplace( opt_dist_tns_index.second_ );

    return nested_anycoord_map_to_nested_nc_pair_array(
        spatial_manager_->get_nodes(
            opt_dtns_idx,
            mpstruct_to_mask_params( mask_params )
        ),
        gc_
    );
}


TiledNodeSequencePairArray*
CAPI::get_distributed_node_sequences(
    const std::size_t& dist_tns_index
)
{
    if ( spatial_manager_ == nullptr )
        throw std::runtime_error( "Spatial grid not initialized yet" );

    return dist_tns_map_to_tns_pair_array(
        spatial_manager_->get_distributed_node_sequence( dist_tns_index ),
        gc_
    );
}


RemoteConnectionInfoPair*
CAPI::get_spatial_connections(
    const std::size_t& conn_map_idx
)
{
    if ( spatial_manager_ == nullptr )
        throw std::runtime_error( "Spatial grid not initialized yet" );

    return rci_to_rcipair(
        spatial_manager_->get_connection_map( conn_map_idx ),
        gc_
    );
}


GridTileVerticesPairArray*
CAPI::get_grid_vertices()
{
    if ( spatial_manager_ == nullptr )
        throw std::runtime_error( "Spatial grid not initialized yet" );

    return gtv_map_to_gtv_pair_array(
        spatial_manager_->get_grid_vertices(),
        gc_
    );
}


RecordedTimesArrayPair*
CAPI::get_timer_data()
{
    if ( spatial_manager_ == nullptr )
        throw std::runtime_error( "Spatial grid not initialized yet" );

    return recorded_times_to_array_pair(
        spatial_manager_->get_timer_data(),
        gc_
    );
}
}
