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
    if ( tci.total_generated_connections_ < 1 )
        return;

    std::unordered_map< conn_index_t, std::size_t >
        source_split_indexes;

    std::unordered_map< std::size_t,
        std::unordered_map< conn_index_t,
        std::tuple< conn_index_t, conn_param_t, conn_param_t > > >
        source_split_connection_maps;

    source_split_connection_maps.emplace(
        std::make_pair(
            0,
            std::unordered_map< conn_index_t,
            std::tuple< conn_index_t, conn_param_t, conn_param_t > >()
        )
    );

    bool at_least_one_emplaced = false;
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

            std::size_t next_map_index = 0;
            auto conn_info = std::make_tuple( pool_index, weight, delay );
            auto driver_search = source_split_indexes.find( driver_index );
            if ( driver_search == source_split_indexes.end() )
            {
                driver_search = source_split_indexes.emplace(
                    std::make_pair(
                        conn_index_t( driver_index ),
                        0
                    )
                ).first;
            }
            else
                next_map_index = driver_search->second;

            for ( mult_t mult = 0; mult < multiplicity; ++mult )
            {
                const auto one_after_last_index = source_split_connection_maps.size();
                if ( next_map_index < one_after_last_index )
                {
                    const auto emplace_it = source_split_connection_maps[ next_map_index ].emplace(
                        std::make_pair(
                            conn_index_t( driver_index ),
                            std::move( conn_info )
                        )
                    );
                    assert( emplace_it.second );
                }
                else
                {
                    std::unordered_map< conn_index_t,
                        std::tuple< conn_index_t, conn_param_t, conn_param_t > > next_map;
                    next_map.emplace(
                        std::make_pair(
                            conn_index_t( driver_index ),
                            std::move( conn_info )
                        )
                    );

                    const auto emplace_it = source_split_connection_maps.emplace(
                        std::make_pair(
                            std::size_t( one_after_last_index ),
                            std::move( next_map )
                        )
                    );
                    assert( emplace_it.second );
                }
                ++next_map_index;
            }

            driver_search->second = next_map_index;
            at_least_one_emplaced = true;
        }
    }

    assert( at_least_one_emplaced );

    count_t emplaced_connections = 0;
    tci.source_unique_connection_vectors_.reserve( source_split_connection_maps.size() );
    for ( const auto& split_conn_map : source_split_connection_maps )
    {
        assert( !split_conn_map.second.empty() );
        ConnectionVectors cvec;
        // Conversion guaranteed by total_generated_connections being positive
        cvec.prepare_vectors( static_cast< count_t >( split_conn_map.second.size() ) );
        emplaced_connections += cvec.sizes_;
        for ( const auto& [driver_index, conn_tuple] : split_conn_map.second )
        {
            const auto& [pool_index, weight, delay] = conn_tuple;

            cvec.connection_sources_.emplace_back( driver_index );
            cvec.connection_targets_.emplace_back( pool_index );
            cvec.connection_weights_.emplace_back( weight );
            cvec.connection_delays_.emplace_back( delay );
        }
        tci.source_unique_connection_vectors_.emplace_back( std::move( cvec ) );
    }

    assert( emplaced_connections == tci.total_generated_connections_ );
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
