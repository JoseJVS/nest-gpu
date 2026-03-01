#ifndef CONNECTION_GENERATION_H
#define CONNECTION_GENERATION_H

#include "random_manager.h"
#include "thread_aligned_array.h"
#include "connection_generators.h"


namespace sapi
{
inline void consolidate_connection_map(
    TileConnectionInfo& tci
)
{
    tci.prepare_vectors();

    auto ci_fl_move_it = std::make_move_iterator( tci.procedural_connection_list_.begin() );
    while ( !tci.procedural_connection_list_.empty() )
    {
        auto ci_fl = *ci_fl_move_it++;
        tci.procedural_connection_list_.pop_front();

        auto ci_move_it = std::make_move_iterator( ci_fl.begin() );
        while ( !ci_fl.empty() )
        {
            const auto [driver_index, pool_index, weight, delay, multiplicity] = *ci_move_it++;
            ci_fl.pop_front();

            for ( mult_t mult = 0; mult < multiplicity; ++mult )
            {
                tci.connection_sources_.emplace_back( driver_index );
                tci.connection_targets_.emplace_back( pool_index );
                tci.connection_weights_.emplace_back( weight );
                tci.connection_delays_.emplace_back( delay );
            }
        }
    }

    // Check correctness
    assert( static_cast< std::size_t >( tci.total_generated_connections_ ) == tci.connection_sources_.size() );
}


template < typename CoordT >
void compute_connections_across_tiles(
    TileConnectionInfo& tci,
    RankPairInfo< CoordT >& rpi,
    const vp_t& target_rank,
    const TAArray< ConnectionGenerator< CoordT > >& cg_array,
    const RandomManager& rng_manager,
    const bool& inverted_pool_driver,
    const bool& allow_multiplicity,
    const bool& allow_self_connections
)
{
    assert(
        rng_manager.is_initialized() &&
        cg_array.is_initialized() &&
        tci.procedural_connection_list_.empty()
    );

    if ( rpi.displacement_checks_map_.consolidated_info_map_.empty() )
        return;

#pragma omp taskgroup
    for ( auto tile_dc_map_pair_it = rpi.displacement_checks_map_.consolidated_info_map_.begin();
        tile_dc_map_pair_it != rpi.displacement_checks_map_.consolidated_info_map_.end();
        ++tile_dc_map_pair_it )
    {
        // At this point there should be no sparsity in the map
        assert( !tile_dc_map_pair_it->second.empty() );

        for ( auto leaf_dc_map_pair_it = tile_dc_map_pair_it->second.begin();
            leaf_dc_map_pair_it != tile_dc_map_pair_it->second.end();
            ++leaf_dc_map_pair_it )
        {
            // At this point there should be no sparsity in the map
            assert( !leaf_dc_map_pair_it->second.empty() );

            tci.procedural_connection_list_.emplace_front();
            const auto leaf_conn_list_it = tci.procedural_connection_list_.begin();

#pragma omp task default( none ) shared( tci, rng_manager, cg_array )\
firstprivate( leaf_conn_list_it, leaf_dc_map_pair_it, tile_dc_map_pair_it,\
    target_rank, inverted_pool_driver, allow_multiplicity, allow_self_connections )
            {
                const auto tid = get_thread_num();
                const auto rng = rng_manager.reseed_rank_paired_rng(
                    tid, target_rank, inverted_pool_driver,
                    tile_dc_map_pair_it->first, leaf_dc_map_pair_it->first
                );
                cg_array.get_thread_item( tid )->compute_connections(
                    tci, *leaf_conn_list_it, leaf_dc_map_pair_it->second,
                    rng, allow_multiplicity, allow_self_connections
                );
            }
        }
    }

    // Cleanup
    rpi.displacement_checks_map_.consolidated_info_map_.clear();

    consolidate_connection_map( tci );
}
}


#endif
