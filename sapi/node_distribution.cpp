/*
 *  node_distribution.cpp
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

#include "grid_neighborhood.h"
#include "node_distribution.h"


namespace sapi
{
void aggregate_tiled_node_count_by_rank(
    NodeCountVector& node_counts_per_rank,
    TileIdxNodeCountPairListVector& tiled_node_counts_per_rank,
    const tileidx_t& tile_index,
    const nodeidx_t& node_count_in_tile,
    const GridNeighborhood& grid_neighborhood,
    const RandomManager& rng_manager,
    const bool& balanced
)
{
    assert( 0 <= node_count_in_tile );

    if ( node_count_in_tile == 0 )
        return;

    const auto tile_owners = grid_neighborhood.tile_ranks_ownership_map_.cbegin() + tile_index;
    assert( tile_owners != grid_neighborhood.tile_ranks_ownership_map_.end() );
    if ( tile_owners->empty() )
        throw std::runtime_error( "Tile with allocated nodes has no owning rank" );

    assert( tile_owners->size() < std::numeric_limits< tileidx_t >::max() );

    const auto node_counts_per_owning_rank = uniform_distribute_node_counts(
        node_count_in_tile,
        static_cast< tileidx_t >( tile_owners->size() ),
        rng_manager,
        balanced,
        true // global
    );

    auto nc_it = node_counts_per_owning_rank.cbegin();
    for ( const auto& owner_rank : *tile_owners )
    {
        const auto count = *nc_it++;
        if ( count == 0 ) continue;
        node_counts_per_rank[ owner_rank ] += count;
        tiled_node_counts_per_rank[ owner_rank ].emplace_front(
            std::make_pair( tile_index, count )
        );
    }
}


DistributedTiledNodeSequenceMap
consolidate_node_sequences_per_tile_per_rank(
    const RankNodeSequenceMap& node_seq_per_rank,
    const TileIdxNodeCountPairListVector& node_counts_per_tile_per_rank
)
{
    nodeidx_t first_node_idx;
    DistributedTiledNodeSequenceMap dist_tns;
    for ( const auto& [rank, node_sequence] : node_seq_per_rank )
    {
        if ( node_sequence.first < 0 ||
            node_sequence.second < 1 )
            throw std::invalid_argument( "Invalid node sequence" );

        first_node_idx = node_sequence.first;
        TileIdxNodeSequenceMap tile_idx_node_sequence_map;
        for ( const auto& [tile_index, node_count]
            : node_counts_per_tile_per_rank.at( rank ) )
        {
            if ( node_count < 1 )
                continue;

            const auto emplace_res = tile_idx_node_sequence_map.emplace(
                std::make_pair(
                    tileidx_t( tile_index ),
                    NodeSequence(
                        first_node_idx,
                        node_count
                    )
                )
            );
            assert( emplace_res.second );

            first_node_idx += node_count;
        }
        assert( ( first_node_idx - node_sequence.first ) == node_sequence.second );

        if ( !tile_idx_node_sequence_map.empty() )
            dist_tns.emplace(
                std::make_pair(
                    vp_t( rank ),
                    std::move( tile_idx_node_sequence_map )
                )
            );
    }

    return dist_tns;
}
}
