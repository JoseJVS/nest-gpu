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
    for ( auto& [tile_index, tile_lci_map] : tci.aggregated_connection_map_ )
    {
        assert( !tile_lci_map.empty() );

        for ( auto& [leaf_index, lci] : tile_lci_map )
        {
            if ( lci.connection_map_.empty() )
                continue;

            tci.total_generated_connections_ += lci.total_generated_connections_;

            for ( auto& [driver_index, pool_conns] : lci.connection_map_ )
            {
                assert( !pool_conns.empty() );

                const auto emplace_res = tci.consolidated_connection_map_.emplace(
                    std::make_pair(
                        nodeidx_t( driver_index ),
                        std::move( pool_conns )
                    )
                );
                assert( emplace_res.second );
            }

            lci.connection_map_.clear();
        }

        tile_lci_map.clear();
    }

    tci.aggregated_connection_map_.clear();

    // Check overflow
    assert( 0 <= tci.total_generated_connections_ &&
        ( ( tci.total_generated_connections_ == 0 ) == tci.consolidated_connection_map_.empty() ) );
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
        tci.aggregated_connection_map_.empty()
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

        const auto tile_conn_map_it = tci.aggregated_connection_map_.emplace(
            std::make_pair(
                tileidx_t( tile_dc_map_pair_it->first ),
                std::unordered_map< tileidx_t, LeafConnectionInfo >()
            )
        ).first;

        for ( auto leaf_dc_map_pair_it = tile_dc_map_pair_it->second.begin();
            leaf_dc_map_pair_it != tile_dc_map_pair_it->second.end();
            ++leaf_dc_map_pair_it )
        {
            // At this point there should be no sparsity in the map
            assert( !leaf_dc_map_pair_it->second.empty() );

            const auto leaf_conn_map_it = tile_conn_map_it->second.emplace(
                std::make_pair(
                    tileidx_t( leaf_dc_map_pair_it->first ),
                    LeafConnectionInfo()
                )
            ).first;

#pragma omp task default( none ) shared( rng_manager, cg_array )\
firstprivate( leaf_conn_map_it, leaf_dc_map_pair_it, tile_dc_map_pair_it,\
    target_rank, inverted_pool_driver, allow_multiplicity, allow_self_connections )
            {
                const auto tid = get_thread_num();
                const auto rng = rng_manager.reseed_rank_paired_rng(
                    tid, target_rank, inverted_pool_driver,
                    tile_dc_map_pair_it->first, leaf_dc_map_pair_it->first
                );
                cg_array.get_thread_item( tid )->compute_connections(
                    leaf_conn_map_it->second, leaf_dc_map_pair_it->second,
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
