/*
 *  node_creation.h
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

#ifndef NODE_CREATION_H
#define NODE_CREATION_H

#include "node_collection.h"
#include "node_distribution.h"


namespace sapi
{
enum DISTRIBUTION_MODE
{
    FREE,
    SQUEEZED,
    BALANCED
};


template < typename I,
    typename std::enable_if_t< std::is_integral_v< rcvref< I > >, bool > = true >
constexpr DISTRIBUTION_MODE
get_distribution_mode( I&& mode )
{
    switch ( mode )
    {
    case 0:
        return DISTRIBUTION_MODE::FREE;
    case 1:
        return DISTRIBUTION_MODE::SQUEEZED;
    case 2:
        return DISTRIBUTION_MODE::BALANCED;
    default:
        throw std::invalid_argument( "Invalid distribution mode" );
    }
}


// Here both returned vectors are in size == num processes.
// The first vector contains the total number of nodes
// to be instantiated in each rank.
// The second vector contains the total number of nodes
// to be instantiated per tile owned by each rank.
template <
    typename I,
    typename CoordT,
    typename std::enable_if_t< std::is_integral_v< rcvref< I > >, bool > = true
>
std::pair< NodeCountVector, TileIdxNodeCountPairListVector >
distribute_node_counts_in_grid(
    const I& num_nodes,
    const std::optional< std::set< tileidx_t > >& tile_set,
    const TileGrid< CoordT >& tile_grid,
    const GridNeighborhood& grid_neighborhood,
    const RandomManager& rng_manager,
    const uint8_t& mode_int
)
{
    assert(
        !tile_grid.positions_.empty() &&
        grid_neighborhood.has_owners_ &&
        rng_manager.is_initialized()
    );

    const auto mode = get_distribution_mode( mode_int );
    const auto large_distribution = std::numeric_limits< nodeidx_t >::max() < num_nodes;

    if ( num_nodes < 0 )
        throw std::invalid_argument(
            "Cannot distribute a negative number of nodes"
        );
    if ( std::numeric_limits< largenodeidx_t >::max() < num_nodes )
        throw std::invalid_argument(
            "Cannot distribute more than " + std::to_string( std::numeric_limits< largenodeidx_t >::max() ) + " nodes"
        );
    if ( large_distribution && tile_set.has_value() )
        throw std::invalid_argument(
            "Distribution of a large number of nodes can only be done over all tiles"
        );
    if ( large_distribution && mode != grid_neighborhood.rank_tile_bijection_ )
        throw std::invalid_argument(
            "Distribution of a large number of nodes can only be done over a bijected grid"
        );
    if ( large_distribution && mode != DISTRIBUTION_MODE::BALANCED )
        throw std::invalid_argument(
            "Distribution of a large number of nodes can only be done using balanced distribution mode"
        );
    if ( tile_set.has_value() && mode == DISTRIBUTION_MODE::FREE )
        throw std::invalid_argument(
            "Tile specific distribution of nodes cannot be done using free distribution mode"
        );

    NodeCountVector node_counts_per_rank( grid_neighborhood.num_processes_, 0 );
    TileIdxNodeCountPairListVector tiled_node_counts_per_rank( grid_neighborhood.num_processes_ );

    if ( num_nodes == 0 )
        return std::make_pair(
            std::move( node_counts_per_rank ), std::move( tiled_node_counts_per_rank )
        );

#pragma omp parallel default( none )\
shared( tile_set, tile_grid, grid_neighborhood, rng_manager,\
    node_counts_per_rank, tiled_node_counts_per_rank )\
firstprivate( num_nodes, large_distribution, mode )
#pragma omp master
#pragma omp taskgroup
    {
        const auto balanced = mode == DISTRIBUTION_MODE::BALANCED;
        const auto node_counts_per_tile = large_distribution
            ? uniform_distribute_node_counts(
                static_cast< largenodeidx_t >( num_nodes ),
                tile_grid.num_tiles_,
                rng_manager,
                true, // balanced
                true // global
            )
            : tile_set.has_value()
            ? uniform_distribute_node_counts(
                static_cast< nodeidx_t >( num_nodes ),
                //Given that tiles are uniquely indexed with tileidx_t
                // the size of tile set will always fit in tileidx_t
                static_cast< tileidx_t >( tile_set.value().size() ),
                rng_manager,
                balanced,
                true // global
            )
            : mode == DISTRIBUTION_MODE::FREE
            ? uniform_distribute_node_counts(
                static_cast< nodeidx_t >( num_nodes ),
                tile_grid.num_tiles_,
                tile_grid.dimensions_,
                rng_manager,
                true // global
            )
            : uniform_distribute_node_counts(
                static_cast< nodeidx_t >( num_nodes ),
                tile_grid.num_tiles_,
                rng_manager,
                balanced,
                true // global
            );

        if ( grid_neighborhood.rank_tile_bijection_ && !tile_set.has_value() )
        {
            for ( vp_t rank = 0; rank < grid_neighborhood.num_processes_; ++rank )
            {
                const auto owned_tile = *grid_neighborhood.rank_tiles_ownership_map_[ rank ].cbegin();
                const auto tile_counts = node_counts_per_tile[ owned_tile ];
                node_counts_per_rank[ rank ] = tile_counts;
                tiled_node_counts_per_rank[ rank ].emplace_front(
                    std::make_pair( owned_tile, tile_counts )
                );
            }
        }
        else if ( tile_set.has_value() )
        {
            auto nc_it = node_counts_per_tile.cbegin();
            for ( const auto& tile_index : tile_set.value() )
                aggregate_tiled_node_count_by_rank(
                    node_counts_per_rank,
                    tiled_node_counts_per_rank,
                    tile_index,
                    *nc_it++,
                    grid_neighborhood,
                    rng_manager,
                    balanced
                );
        }
        else {
            auto nc_it = node_counts_per_tile.cbegin();
            for ( tileidx_t tile_index = 0; tile_index < tile_grid.num_tiles_; ++tile_index )
                aggregate_tiled_node_count_by_rank(
                    node_counts_per_rank,
                    tiled_node_counts_per_rank,
                    tile_index,
                    *nc_it++,
                    grid_neighborhood,
                    rng_manager,
                    balanced
                );
        }
    }

    return std::make_pair(
        std::move( node_counts_per_rank ), std::move( tiled_node_counts_per_rank )
    );
}


template < typename CoordT >
void distribute_node_counts_in_tiles(
    GridNodeCollection< CoordT >& grid_node_col,
    const TileIdxNodeSequenceMap& node_seq_per_tile,
    const TileGrid< CoordT >& tile_grid,
    const RandomManager& rng_manager,
    const uint8_t& mode_int
)
{
    assert(
        !grid_node_col.tiles_node_coord_map_.empty() &&
        tile_grid.has_split_ &&
        rng_manager.is_initialized()
    );

    if ( node_seq_per_tile.empty() )
        throw std::invalid_argument( "Cannot generate nodes with empty node sequences" );

    const auto mode = get_distribution_mode( mode_int );

#pragma omp parallel default( none )\
shared( node_seq_per_tile, grid_node_col, tile_grid, rng_manager, mode )
#pragma omp master
#pragma omp taskgroup
    {
        const auto free = mode == DISTRIBUTION_MODE::FREE;
        const auto balanced = mode == DISTRIBUTION_MODE::BALANCED;
        for ( const auto& [tile_index, node_sequence] : node_seq_per_tile )
        {
            assert( 0 <= node_sequence.first && 0 < node_sequence.second );

            const auto tile_nc_it = grid_node_col.tiles_node_coord_map_.find( tile_index );
            assert( tile_nc_it != grid_node_col.tiles_node_coord_map_.end() );
            const auto tile_pos_it = tile_grid.positions_.cbegin() + tile_index;

            const auto node_count_per_sub_tile = free
                ? uniform_distribute_node_counts(
                    node_sequence.second,
                    static_cast< tileidx_t >( tile_nc_it->second.sub_tiles_vector_.size() ),
                    tile_pos_it->get_tile()->get_possible_sub_tile_branches( tile_grid.splits_ ),
                    rng_manager,
                    false // global
                )
                : uniform_distribute_node_counts(
                    node_sequence.second,
                    static_cast< tileidx_t >( tile_nc_it->second.sub_tiles_vector_.size() ),
                    rng_manager,
                    balanced,
                    false // global
                );

            nodeidx_t st_node_index = node_sequence.first;
            auto st_it = tile_nc_it->second.sub_tiles_vector_.cbegin();
            for ( const auto& node_count : node_count_per_sub_tile )
            {
                if ( node_count < 1 )
                {
                    ++st_it;
                    continue;
                }

                const auto sub_tile = *st_it++;
                const auto coord_map_it = tile_nc_it->second.sub_tiles_node_coord_map_.find(
                    sub_tile->index_
                );
                assert( coord_map_it != tile_nc_it->second.sub_tiles_node_coord_map_.end() );

#pragma omp task default( none ) shared( rng_manager )\
firstprivate( sub_tile, st_node_index, node_count, coord_map_it, tile_index )
                {
                    const auto tid = get_thread_num();
                    const auto rng = rng_manager.reseed_rank_paired_rng(
                        tid, rng_manager.local_rank_, false,
                        tile_index, sub_tile->index_
                    );
                    const auto coord_map_emplace_res =
                        coord_map_it->second.emplace(
                            std::make_pair(
                                nodeidx_t( st_node_index ),
                                std::vector< CoordT >( node_count )
                            )
                        );
                    assert( coord_map_emplace_res.second );
                    sub_tile->generate_coords_in_tile( coord_map_emplace_res.first->second, rng );
                }

                st_node_index += node_count;
            }
            assert( ( st_node_index - node_sequence.first ) == node_sequence.second );
        }
    }
}
}


#endif
