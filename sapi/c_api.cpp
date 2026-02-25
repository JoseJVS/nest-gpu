#include "c_api.h"
#include "api_converters.h"


void sapi::CAPI::reset()
{
    gc_.free_gc();
    spatial_manager_.release();
    seed_ = DEFAULT_BASE_SEED_;
    rng_type_ = DEFAULT_RNG_TYPE_;
}


void sapi::CAPI::free_gc()
{
    gc_.free_gc();
}


sapi::vp_t sapi::CAPI::get_rank() const
{
    if ( !is_initialized() )
        throw std::runtime_error( "API not initialized yet" );

    return local_rank_;
}


void sapi::CAPI::set_rank( const vp_t& rank )
{
    if ( is_initialized() )
        throw std::runtime_error( "API is already initialized" );
    if ( rank < 0 )
        throw std::runtime_error( "Invalid MPI rank" );

    local_rank_ = rank;
}


sapi::vp_t sapi::CAPI::get_num_processes() const
{
    if ( !is_initialized() )
        throw std::runtime_error( "API not initialized yet" );

    return num_processes_;
}


void sapi::CAPI::set_num_processes( const vp_t& num_processes )
{
    if ( is_initialized() )
        throw std::runtime_error( "API is already initialized" );
    if ( num_processes < 1 )
        throw std::runtime_error( "Invalid MPI num processes" );

    num_processes_ = num_processes;
}


sapi::vp_t sapi::CAPI::get_num_threads() const
{
    if ( !is_initialized() )
        throw std::runtime_error( "API not initialized yet" );

    return get_max_omp_threads();
}


void sapi::CAPI::set_num_threads( const vp_t& num_threads )
{
    if ( !is_initialized() )
        throw std::runtime_error( "API not initialized yet" );

    set_max_omp_threads( num_threads );
    if ( manager_initialized() )
        spatial_manager_->update_num_threads();
}

uint32_t sapi::CAPI::get_rng_seed() const
{
    return seed_;
}


void sapi::CAPI::set_rng_seed( const uint32_t& seed )
{
    seed_ = seed;
    if ( manager_initialized() )
        spatial_manager_->set_rng_seed( seed );
}


sapi::CharArray* sapi::CAPI::get_rng_type()
{
    return string_to_charray( rng_type_, gc_ );
}


void sapi::CAPI::set_rng_type( const CharArray& rng_type )
{
    rng_type_ = charray_to_string( rng_type );
    if ( manager_initialized() )
        spatial_manager_->set_rng_type( rng_type_ );
}


void sapi::CAPI::generate_tile_grid(
    const SpaceTArray& grid_origin,
    const TileIdxArray& grid_dimensions,
    const CharArray& tile_type,
    const SpaceTArray& tile_params,
    const NestedTileIdxArray& rank_tiles_ownership_map,
    const split_t& num_splits,
    const bool& edge_wrap
)
{
    if ( !is_initialized() )
        throw std::runtime_error( "API not initialized yet" );

    if ( manager_initialized() )
        throw std::runtime_error( "Cannot initialize spatial manager more than once" );

    switch ( grid_dimensions.size_ )
    {
    case 2:
        spatial_manager_ = std::make_unique< SpatialManager< Coord2D > >(
            local_rank_, num_processes_
        );
        break;

    case 3:
        spatial_manager_ = std::make_unique< SpatialManager< Coord3D > >(
            local_rank_, num_processes_
        );

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
        array_to_vector( tile_params ),
        nested_array_to_set_vector( rank_tiles_ownership_map ),
        num_splits,
        edge_wrap
    );
}


sapi::NodeCountVector
sapi::CAPI::generate_nodes_in_grid(
    const largenodeidx_t& num_nodes,
    const TileIdxArray& tile_set,
    const uint8_t& distribution_mode
)
{
    if ( !manager_initialized() )
        throw std::runtime_error( "Spatial grid not initialized yet" );

    std::optional< std::set< tileidx_t > > set;
    if ( tile_set.size_ > 0 )
        set.emplace( array_to_set( tile_set ) );

    return spatial_manager_->distribute_nodes_in_grid(
        num_nodes,
        set,
        distribution_mode
    );
}


std::size_t sapi::CAPI::generate_nodes_in_tiles(
    const RankNodeSequenceMap& node_sequence_map,
    const uint8_t& distribution_mode
)
{
    if ( !manager_initialized() )
        throw std::runtime_error( "Spatial grid not initialized yet" );

    return spatial_manager_->generate_nodes_in_tiles(
        node_sequence_map,
        distribution_mode
    );
}


std::pair< sapi::NodeCountVector, sapi::NestedSpaceTArray* >
sapi::CAPI::insert_positions_in_grid(
    const NestedSpaceTArray& anycoord_array
)
{
    if ( !manager_initialized() )
        throw std::runtime_error( "Spatial grid not initialized yet" );

    auto anycoord_vector = nested_array_to_nested_vector( anycoord_array );

    const auto node_counts_per_rank = spatial_manager_->insert_positions_in_grid( anycoord_vector );

    return std::make_pair(
        node_counts_per_rank,
        nested_collection_to_nested_array( anycoord_vector, gc_ )
    );
}


std::size_t sapi::CAPI::insert_positions_in_tiles(
    const RankNodeSequenceMap& node_sequence_map
)
{
    if ( !manager_initialized() )
        throw std::runtime_error( "Spatial grid not initialized yet" );

    return spatial_manager_->insert_positions_in_tiles(
        node_sequence_map
    );
}


std::pair< std::size_t, sapi::RankConnectionInfo* >
sapi::CAPI::compute_spatial_connections(
    const std::size_t& dist_tns_source_index,
    const std::size_t& dist_tns_target_index,
    const MPStruct& mask_params,
    const CPStruct& conn_params
)
{
    if ( !manager_initialized() )
        throw std::runtime_error( "Spatial grid not initialized yet" );

    return spatial_manager_->compute_spatial_connections(
        dist_tns_source_index,
        dist_tns_target_index,
        mpstruct_to_mask_params( mask_params ),
        cpstruct_to_conn_params( conn_params )
    );
}


sapi::NestedNodeCoordPairArray*
sapi::CAPI::get_nodes(
    const OptionalIndex& opt_dist_tns_index,
    const MPStruct& mask_params
)
{
    if ( !manager_initialized() )
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


sapi::TiledNodeSequencePairArray*
sapi::CAPI::get_distributed_node_sequences(
    const std::size_t& dist_tns_index
)
{
    if ( !manager_initialized() )
        throw std::runtime_error( "Spatial grid not initialized yet" );

    return dist_tns_map_to_tns_pair_array(
        spatial_manager_->get_distributed_node_sequence( dist_tns_index ),
        gc_
    );
}


sapi::RCIStruct*
sapi::CAPI::get_spatial_connections(
    const std::size_t& conn_map_idx
)
{
    if ( !manager_initialized() )
        throw std::runtime_error( "Spatial grid not initialized yet" );

    return rci_to_rcistruct(
        spatial_manager_->get_connection_map( conn_map_idx ),
        gc_
    );
}


sapi::GridTileVerticesPairArray*
sapi::CAPI::get_grid_vertices()
{
    if ( !manager_initialized() )
        throw std::runtime_error( "Spatial grid not initialized yet" );

    return gtv_map_to_gtv_pair_array(
        spatial_manager_->get_grid_vertices(),
        gc_
    );
}


sapi::TimerDataPairArray*
sapi::CAPI::get_timer_data()
{
    if ( !manager_initialized() )
        throw std::runtime_error( "Spatial grid not initialized yet" );

    return timer_data_map_to_timer_data_pair_array(
        spatial_manager_->get_timer_data(),
        gc_
    );
}
