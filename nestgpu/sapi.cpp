#include <iostream>

#include "sapi.h"
#include "c_api.h"


#define START_TRY_BOOL try {
#define END_TRY_BOOL return true; } catch( const std::exception& e ) {\
    std::cerr << e.what() << '\n'; return false; }

#define START_TRY_OPT OptionalIndex opt; try { opt.first_ = true;
#define END_TRY_OPT return opt; } catch( const std::exception& e ) {\
    std::cerr << e.what() << '\n'; opt.first_ = false; return opt; }

#define START_TRY_PTR try {
#define END_TRY_PTR } catch( const std::exception& e ) {\
    std::cerr << e.what() << '\n'; return nullptr; }


namespace sapi
{
struct MPIFinalizer
{
    bool initialized_ = false;
    ~MPIFinalizer()
    {
        finalize_mpi();
    }
};


extern "C"
{
    static CAPI capi;
    static MPIFinalizer mpif;
    static NodeCountVector total_nodes_per_rank;

    RankNodeSequenceMap
        update_node_counts_per_rank(
            const NodeCountVector& node_counts_per_rank
        )
    {
        vp_t rank = 0;
        RankNodeSequenceMap rns_map;
        auto nc_it = node_counts_per_rank.cbegin();
        for ( auto& rank_node_count : total_nodes_per_rank )
        {
            if ( *nc_it < 1 )
            {
                ++rank;
                ++nc_it;
                continue;
            }

            if (
                !rns_map.emplace(
                    std::make_pair(
                        rank++,
                        NodeSequence( rank_node_count, *nc_it )
                    )
                ).second
                )
                throw std::runtime_error( "Corrupted rank node sequence map" );

            rank_node_count += *nc_it++;
        }

        return rns_map;
    }

    bool init( vp_t argc, char** argv )
    {
        START_TRY_BOOL
            if ( mpif.initialized_ )
                throw std::runtime_error( "Cannot initialize multiple times" );

        init_mpi( &argc, &argv );
        capi.set_rank( get_mpi_rank() );
        const auto  num_procs = get_num_mpi_processes();
        capi.set_num_processes( num_procs );
        total_nodes_per_rank.resize( num_procs, 0 );
        mpif.initialized_ = true;
        END_TRY_BOOL
    }

    bool reset_api()
    {
        START_TRY_BOOL
            capi.reset();
        total_nodes_per_rank.clear();
        total_nodes_per_rank.resize( capi.get_num_processes(), 0 );
        END_TRY_BOOL
    }

    bool free_gc()
    {
        START_TRY_BOOL
            capi.free_gc();
        END_TRY_BOOL
    }

    OptionalIndex get_rank()
    {
        START_TRY_OPT
            opt.second_ = static_cast< std::size_t >( capi.get_rank() );
        END_TRY_OPT
    }

    OptionalIndex get_num_processes()
    {
        START_TRY_OPT
            opt.second_ = static_cast< std::size_t >( capi.get_num_processes() );
        END_TRY_OPT
    }

    OptionalIndex get_num_threads()
    {
        START_TRY_OPT
            opt.second_ = static_cast< std::size_t >( capi.get_num_threads() );
        END_TRY_OPT
    }

    bool set_num_threads( const vp_t& num_threads )
    {
        START_TRY_BOOL
            capi.set_num_threads( num_threads );
        END_TRY_BOOL
    }

    OptionalIndex get_rng_seed()
    {
        START_TRY_OPT
            opt.second_ = static_cast< std::size_t >( capi.get_rng_seed() );
        END_TRY_OPT
    }

    bool set_rng_seed( const uint32_t& seed )
    {
        START_TRY_BOOL
            capi.set_rng_seed( seed );
        END_TRY_BOOL
    }

    CharArray* get_rng_type()
    {
        START_TRY_PTR
            return capi.get_rng_type();
        END_TRY_PTR
    }

    bool set_rng_type( const CharArray& rng_type )
    {
        START_TRY_BOOL
            capi.set_rng_type( rng_type );
        END_TRY_BOOL
    }

    bool generate_tile_grid(
        const SpaceTArray& grid_origin,
        const TileIdxArray& grid_dimensions,
        const CharArray& tile_type,
        const SpaceTArray& tile_params,
        const NestedTileIdxArray& rank_tiles_ownership_map,
        const split_t& num_splits,
        const bool& edge_wrap
    )
    {
        START_TRY_BOOL
            capi.generate_tile_grid(
                grid_origin,
                grid_dimensions,
                tile_type,
                tile_params,
                rank_tiles_ownership_map,
                num_splits,
                edge_wrap
            );
        END_TRY_BOOL
    }

    OptionalIndex generate_nodes_in_grid(
        const largenodeidx_t& num_nodes,
        const TileIdxArray& tile_set,
        const uint8_t& grid_distribution_mode,
        const uint8_t& tile_distribution_mode
    )
    {
        START_TRY_OPT
            const auto nodes_per_rank = capi.generate_nodes_in_grid(
                num_nodes,
                tile_set,
                grid_distribution_mode
            );

        opt.second_ = capi.generate_nodes_in_tiles(
            update_node_counts_per_rank( nodes_per_rank ),
            tile_distribution_mode
        );
        END_TRY_OPT
    }

    PairT< OptionalIndex, NestedSpaceTArray* >
        insert_positions_in_grid(
            const NestedSpaceTArray& anycoord_array
        )
    {
        PairT< OptionalIndex, NestedSpaceTArray* > pair;
        try
        {
            const auto [
                nodes_per_rank,
                leftovers
            ] = capi.insert_positions_in_grid(
                anycoord_array
            );

            pair.first_.first_ = true;
            pair.first_.second_ = capi.insert_positions_in_tiles(
                update_node_counts_per_rank( nodes_per_rank )
            );
            pair.second_ = leftovers;
            return pair;
        }
        catch ( const std::exception& e )
        {
            std::cerr << e.what() << '\n';
            pair.first_.first_ = false;
            pair.second_ = nullptr;
            return pair;
        }
    }

    OptionalIndex compute_spatial_connections(
        const std::size_t& dist_tns_source_index,
        const std::size_t& dist_tns_target_index,
        const MPStruct& mask_params,
        const CPStruct& conn_params
    )
    {
        START_TRY_OPT
            opt.second_ = capi.compute_spatial_connections(
                dist_tns_source_index,
                dist_tns_target_index,
                mask_params,
                conn_params
            ).first;
        END_TRY_OPT
    }

    NestedNodeCoordPairArray* get_nodes(
        const PairT< bool, std::size_t >& opt_dist_tns_index,
        const MPStruct& mask_params
    )
    {
        START_TRY_PTR
            return capi.get_nodes( opt_dist_tns_index, mask_params );
        END_TRY_PTR
    }

    TiledNodeSequencePairArray* get_distributed_node_sequences(
        const std::size_t& dist_tns_index
    )
    {
        START_TRY_PTR
            return capi.get_distributed_node_sequences(
                dist_tns_index
            );
        END_TRY_PTR
    }

    RCIStruct* get_spatial_connections(
        const std::size_t& conn_idx
    )
    {
        START_TRY_PTR
            return capi.get_spatial_connections(
                conn_idx
            );
        END_TRY_PTR
    }

    GridTileVerticesPairArray* get_grid_vertices()
    {
        START_TRY_PTR
            return capi.get_grid_vertices();
        END_TRY_PTR
    }

    TimerDataPairArray* get_timer_data()
    {
        START_TRY_PTR
            return capi.get_timer_data();
        END_TRY_PTR
    }
}
}
