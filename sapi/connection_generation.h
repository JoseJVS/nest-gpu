/*
 *  connection_generation.h
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

#ifndef CONNECTION_GENERATION_H
#define CONNECTION_GENERATION_H

#include "random_manager.h"
#include "connection_generator.h"


namespace sapi
{
void consolidate_connection_map(
    TileConnectionInfo& tci,
    const bool& partition_connections_by_source
);


template < typename CoordT >
void compute_connections_across_tiles(
    TileConnectionInfo& tci,
    RankPairInfo< CoordT >& rpi,
    const vp_t& target_rank,
    const TAArray< ConnectionGenerator >& cg_array,
    const RandomManager& rng_manager,
    const bool& inverted_pool_driver,
    const bool& allow_multiplicity,
    const bool& allow_self_connections,
    const bool& partition_connections_by_source
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
                    *rng, allow_multiplicity, allow_self_connections
                );
            }
        }
    }

    // Cleanup
    rpi.displacement_checks_map_.consolidated_info_map_.clear();

    consolidate_connection_map( tci, partition_connections_by_source );
}
}


#endif
