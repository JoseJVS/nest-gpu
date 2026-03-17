/*
 *  node_collection.h
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

#ifndef NODE_COLLECTION_H
#define NODE_COLLECTION_H

#include "tile_grid.h"
#include "node_containers.h"


namespace sapi
{
template < typename CoordT >
struct TileNodeCollection
{
    std::vector< const Tile< CoordT >* > sub_tiles_vector_;
    TileIdxNodeIdxCoordMap< CoordT > sub_tiles_node_coord_map_;

    TileNodeCollection() = default;
    TileNodeCollection( const TileNodeCollection& ) = delete;
    TileNodeCollection( TileNodeCollection&& ) = default;
    ~TileNodeCollection() = default;

    void initialize_maps(
        const TilePosition< CoordT >&,
        const split_t&
    );
};


template < typename CoordT >
void TileNodeCollection< CoordT >::initialize_maps(
    const TilePosition< CoordT >& tile_position,
    const split_t& splits
)
{
    assert(
        sub_tiles_vector_.empty() &&
        sub_tiles_node_coord_map_.empty()
    );
    sub_tiles_vector_ = tile_position.tile_.get_leaf_sub_tiles( splits );
    for ( const auto& st_ptr : sub_tiles_vector_ )
    {
        assert( st_ptr != nullptr );
        const auto emplace_res = sub_tiles_node_coord_map_.emplace(
            std::make_pair(
                tileidx_t( st_ptr->index_ ),
                NodeIdxCoordMap< CoordT >()
            )
        );
        assert( emplace_res.second );
    }
}


template < typename CoordT >
struct GridNodeCollection
{
    std::unordered_map<
        tileidx_t,
        TileNodeCollection< CoordT >
    > tiles_node_coord_map_;

    GridNodeCollection() = default;
    GridNodeCollection( const GridNodeCollection& ) = delete;
    GridNodeCollection( GridNodeCollection&& ) = default;
    ~GridNodeCollection() = default;

    void initialize_map(
        const std::set< tileidx_t >& locally_owned_tiles,
        const TileGrid< CoordT >& tile_grid
    );
};


template < typename CoordT >
void GridNodeCollection< CoordT >::initialize_map(
    const std::set< tileidx_t >& locally_owned_tiles,
    const TileGrid< CoordT >& tile_grid
)
{
    assert( tiles_node_coord_map_.empty() && tile_grid.has_split_ );

    if ( locally_owned_tiles.empty() )
        return;

    for ( const auto& tile_position : locally_owned_tiles )
        tiles_node_coord_map_.emplace(
                std::make_pair(
                    tileidx_t( tile_position ),
                    TileNodeCollection< CoordT >()
                )
        ).first->second.initialize_maps(
            tile_grid.positions_.at( tile_position ),
            tile_grid.splits_
        );
}
}


#endif
