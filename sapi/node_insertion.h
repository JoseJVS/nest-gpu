/*
 *  node_insertion.h
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

#ifndef NODE_INSERTION_H
#define NODE_INSERTION_H

#include <tuple>
#include <iterator>

#include "node_collection.h"
#include "node_distribution.h"


namespace sapi
{
// Forward definition to coordinate_geometry.h
template < typename CoordT >
space_t distance2( const CoordT& coordA, const CoordT& coordB );


template < typename CoordT >
bool recursive_bounds_test(
    CoordT&& coord,
    TiledCoordMap< CoordT >& tile_coord_map,
    const BoundingBox< CoordT >& bounding_box,
    const TileGrid< CoordT >& tile_grid,
    const GridNeighborhood& grid_neighborhood
)
{
    if ( !bounding_box.coord_in_box( coord ) )
        return false;

    bool inserted = false;
    if ( bounding_box.inner_boxes_.empty() )
    {
        assert(
            0 <= bounding_box.tile_index_ &&
            bounding_box.tile_index_ < tile_grid.num_tiles_
        );
        if (
            !grid_neighborhood.tile_ranks_ownership_map_[ bounding_box.tile_index_ ].empty() &&
            tile_grid.positions_[ bounding_box.tile_index_ ].get_tile()->coord_in_tile(
                coord
            ) )
        {
            auto search = tile_coord_map.find( bounding_box.tile_index_ );
            if ( search == tile_coord_map.end() )
                search = tile_coord_map.emplace(
                    std::make_pair(
                        tileidx_t( bounding_box.tile_index_ ),
                        std::list< CoordT >()
                    )
                ).first;

            search->second.emplace_back( std::move( coord ) );
            inserted = true;
        }
    }
    else
    {
        for ( const auto& bb : bounding_box.inner_boxes_ )
        {
            inserted |= recursive_bounds_test(
                std::move( coord ),
                tile_coord_map,
                bb,
                tile_grid,
                grid_neighborhood
            );
            if ( inserted ) break;
        }
    }

    return inserted;
}


template < typename CoordT >
void aggregate_local_node_positions(
    std::list< CoordT >&& coord_list,
    TiledCoordMap< CoordT >& tiled_coord_map,
    const tileidx_t& tile_index,
    const nodeidx_t& move_count,
    const nodeidx_t& skip_count
)
{
    assert( 0 < move_count &&
        0 <= skip_count &&
        static_cast< std::size_t >( skip_count + move_count ) <= coord_list.size()
    );

    // This function should only be called once per tile and only for local tiles
    if ( 0 == skip_count && static_cast< std::size_t >( move_count ) == coord_list.size() )
    {
        tiled_coord_map.emplace(
            std::make_pair(
                tileidx_t( tile_index ),
                std::move( coord_list )
            )
        );
        return;
    }

    const auto emplace_it = tiled_coord_map.emplace(
        std::make_pair(
            tileidx_t( tile_index ),
            std::list< CoordT >()
        )
    ).first;

    nodeidx_t moved_coords = 0;
    nodeidx_t skipped_coords = 0;
    auto coord_move_it = std::make_move_iterator( coord_list.begin() );
    while ( !coord_list.empty() )
    {
        auto coord = *coord_move_it++;
        coord_list.pop_front();

        if ( skipped_coords++ < skip_count )
            continue;

        if ( move_count <= moved_coords++ )
            break;

        emplace_it->second.emplace_back(
            std::move( coord )
        );
    }
}


template < typename CoordT >
void aggregate_tiled_node_count_by_rank(
    TiledCoordMap< CoordT >& global_tiled_coord_map,
    TiledCoordMap< CoordT >& local_tiled_coord_map,
    NodeCountVector& node_counts_per_rank,
    TileIdxNodeCountPairListVector& tiled_node_counts_per_rank,
    const GridNeighborhood& grid_neighborhood,
    const RandomManager& rng_manager
)
{
    assert(
        node_counts_per_rank.size() == static_cast< std::size_t >( grid_neighborhood.num_processes_ ) &&
        tiled_node_counts_per_rank.size() == static_cast< std::size_t >( grid_neighborhood.num_processes_ )
    );

    if ( global_tiled_coord_map.empty() )
        return;

    for ( auto& [tile_index, coord_list] : global_tiled_coord_map )
    {
        assert( !coord_list.empty() &&
            coord_list.size() < std::numeric_limits< nodeidx_t >::max() );

        const auto tile_owners = grid_neighborhood.tile_ranks_ownership_map_.cbegin() + tile_index;
        assert(
            tile_owners != grid_neighborhood.tile_ranks_ownership_map_.end() &&
            !tile_owners->empty() &&
            tile_owners->size() < std::numeric_limits< tileidx_t >::max()
        );

        const auto node_counts_per_owning_rank = uniform_distribute_node_counts(
            static_cast< nodeidx_t >( coord_list.size() ),
            static_cast< tileidx_t >( tile_owners->size() ),
            rng_manager,
            true, // balanced
            true // global
        );

        nodeidx_t coord_skip = 0;
        auto nc_it = node_counts_per_owning_rank.cbegin();
        for ( const auto& owner_rank : *tile_owners )
        {
            const auto count = *nc_it++;
            if ( count == 0 ) continue;
            node_counts_per_rank[ owner_rank ] += count;
            tiled_node_counts_per_rank[ owner_rank ].emplace_front(
                std::make_pair( tile_index, count )
            );

            if ( owner_rank != grid_neighborhood.local_rank_ )
                coord_skip += count;
            else
                // As tile_owners is a set
                // this is guaranteed to happen only once per tile
                aggregate_local_node_positions(
                    std::move( coord_list ),
                    local_tiled_coord_map,
                    tile_index,
                    count,
                    coord_skip
                );
        }

        coord_list.clear();
    }

    global_tiled_coord_map.clear();
}


template < typename CoordT >
std::tuple<
    NodeCountVector, // rank node counts
    TileIdxNodeCountPairListVector, // rank tiled node counts
    TiledCoordMap< CoordT > // sorted coords by locally owned tiles
>
insert_node_positions_in_grid(
    std::list< CoordT >& coord_list,
    const TileGrid< CoordT >& tile_grid,
    const GridNeighborhood& grid_neighborhood,
    const RandomManager& rng_manager
)
{
    assert(
        !tile_grid.positions_.empty() &&
        grid_neighborhood.has_owners_ &&
        rng_manager.is_initialized()
    );

    std::list< CoordT > leftovers;
    NodeCountVector node_counts_per_rank( grid_neighborhood.num_processes_, 0 );
    TileIdxNodeCountPairListVector tiled_node_counts_per_rank( grid_neighborhood.num_processes_ );
    TiledCoordMap< CoordT > local_tiled_coords;

    if ( coord_list.empty() )
        return std::make_tuple(
            std::move( node_counts_per_rank ),
            std::move( tiled_node_counts_per_rank ),
            std::move( local_tiled_coords )
        );

    TiledCoordMap< CoordT > global_tiled_coords;
    auto coord_move_it = std::make_move_iterator( coord_list.begin() );
    while ( !coord_list.empty() )
    {
        auto coord = *coord_move_it++;
        coord_list.pop_front();

        if ( !recursive_bounds_test(
            std::move( coord ),
            global_tiled_coords,
            tile_grid.bounding_box_,
            tile_grid,
            grid_neighborhood
        ) )
        {
            // If test failed it is guaranteed that coord
            // was not moved during recursion
            leftovers.emplace_back( std::move( coord ) );
        }
    }

    aggregate_tiled_node_count_by_rank(
        global_tiled_coords,
        local_tiled_coords,
        node_counts_per_rank,
        tiled_node_counts_per_rank,
        grid_neighborhood,
        rng_manager
    );

    coord_list = std::move( leftovers );

    return std::make_tuple(
        std::move( node_counts_per_rank ),
        std::move( tiled_node_counts_per_rank ),
        std::move( local_tiled_coords )
    );
}


template < typename CoordT >
void recursive_sub_tile_test(
    CoordT&& coord,
    TiledCoordMap< CoordT >& sub_tile_coord_map,
    const Tile< CoordT >* const& tile
)
{
    // Tile coord check is performed at previous recursive call
    // for root tile this is done during recursive bounds test
    if ( !tile->is_split() )
    {
        auto search = sub_tile_coord_map.find( tile->index_ );
        if ( search == sub_tile_coord_map.end() )
            search = sub_tile_coord_map.emplace(
                    std::make_pair(
                        tileidx_t( tile->index_ ),
                        std::list< CoordT >()
                    )
            ).first;

        search->second.emplace_back( std::move( coord ) );
    }
    else
    {
        const Tile< CoordT >* sub_tile = nullptr;
        for ( const auto& st : tile->get_sub_tiles() )
        {
            assert( st );
            if ( st->coord_in_tile( coord ) )
            {
                sub_tile = st.get();
                break;
            }
        }

        if ( sub_tile != nullptr )
            recursive_sub_tile_test(
                std::move( coord ),
                sub_tile_coord_map,
                sub_tile
            );
        else
        {
            space_t min_distance = std::numeric_limits< space_t >::max();
            for ( const auto& st : tile->get_sub_tiles() )
            {
                const auto distance = distance2(
                    coord, st->project_point_to_surface(
                        coord
                    )
                );

                if ( leq_test( distance, min_distance ) )
                {
                    min_distance = distance;
                    sub_tile = st.get();
                }
            }

            assert( sub_tile != nullptr );
            recursive_sub_tile_test(
                std::move( coord ),
                sub_tile_coord_map,
                sub_tile
            );
        }
    }
}


template < typename CoordT >
void insert_node_positions_in_leafs(
    std::list< CoordT >&& coord_list,
    TileNodeCollection< CoordT >& tile_node_col,
    const TilePosition< CoordT >& tile_position,
    const NodeSequence& node_sequence
)
{
    assert( !coord_list.empty() );

    const auto tile = tile_position.get_tile();
    TiledCoordMap< CoordT > sub_tile_coord_map;
    auto coord_move_it = std::make_move_iterator( coord_list.begin() );
    while ( !coord_list.empty() )
    {
        auto coord = *coord_move_it++;
        coord_list.pop_front();

        recursive_sub_tile_test(
            std::move( coord ),
            sub_tile_coord_map,
            tile
        );
    }

    coord_list.clear();
    assert( !sub_tile_coord_map.empty() );

    nodeidx_t st_node_index = node_sequence.first;
    for ( auto& [st_index, st_coord_list] : sub_tile_coord_map )
    {
        assert( !st_coord_list.empty() &&
            st_coord_list.size() < std::numeric_limits< nodeidx_t >::max()
        );

        const auto coord_map_it = tile_node_col.sub_tiles_node_coord_map_.find(
            st_index
        );
        assert( coord_map_it != tile_node_col.sub_tiles_node_coord_map_.end() );

        const auto coord_map_emplace_res = coord_map_it->second.emplace(
            std::make_pair(
                nodeidx_t( st_node_index ),
                std::vector< CoordT >( st_coord_list.size() )
            )
        );
        assert( coord_map_emplace_res.second );

        st_node_index += static_cast< nodeidx_t >( st_coord_list.size() );

        std::move(
            st_coord_list.begin(),
            st_coord_list.end(),
            coord_map_emplace_res.first->second.begin()
        );

        st_coord_list.clear();
    }
    assert( ( st_node_index - node_sequence.first ) == node_sequence.second );

    sub_tile_coord_map.clear();
}


template < typename CoordT >
void insert_node_positions_in_tiles(
    TiledCoordMap< CoordT >&& tiled_coord_map,
    GridNodeCollection< CoordT >& grid_node_col,
    const TileIdxNodeSequenceMap& node_seq_per_tile,
    const TileGrid< CoordT >& tile_grid
)
{
    assert(
        !grid_node_col.tiles_node_coord_map_.empty() &&
        tile_grid.has_split_
    );

    if ( node_seq_per_tile.empty() )
        throw std::invalid_argument( "Cannot generate nodes with empty node sequences" );
    if ( node_seq_per_tile.size() != tiled_coord_map.size() )
        throw std::invalid_argument( "Tiled node sequence mismatch with tiled coord map" );

    for ( const auto& [tile_index, node_sequence] : node_seq_per_tile )
    {
        const auto tcm_it = tiled_coord_map.find( tile_index );
        const auto tnc_it = grid_node_col.tiles_node_coord_map_.find( tile_index );
        assert(
            tcm_it != tiled_coord_map.end() &&
            tnc_it != grid_node_col.tiles_node_coord_map_.end() &&
            0 <= node_sequence.first &&
            0 < node_sequence.second &&
            tcm_it->second.size() == static_cast< std::size_t >( node_sequence.second )
        );

        insert_node_positions_in_leafs(
            std::move( tcm_it->second ),
            tnc_it->second,
            tile_grid.positions_[ tile_index ],
            node_sequence
        );
    }

    tiled_coord_map.clear();
}
}


#endif
