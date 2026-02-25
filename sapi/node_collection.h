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
    sub_tiles_vector_ = tile_position.get_tile()->get_leaf_sub_tiles( splits );
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
    assert(
        tiles_node_coord_map_.empty() &&
        tile_grid.has_split_ &&
        !locally_owned_tiles.empty()
    );

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
