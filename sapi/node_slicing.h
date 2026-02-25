#ifndef NODE_SLICING_H 
#define NODE_SLICING_H 

#include "mask_tile_processing.h"


namespace sapi
{
typedef DistributedTiledNodeSequenceMap::const_iterator DistTns_IT;
typedef TileIdxNodeSequenceMap::const_iterator Tns_IT;


template < typename CoordT >
std::unordered_map< tileidx_t, std::optional< Tns_IT > >
get_optional_tiled_node_sequences(
    const std::optional< DistTns_IT >& opt_dist_tns_it,
    const GridNodeCollection< CoordT >& grid_nc
)
{
    std::unordered_map< tileidx_t, std::optional< Tns_IT > > optional_tns;

    if ( opt_dist_tns_it.has_value() )
    {
        auto tns_it = opt_dist_tns_it.value()->second.cbegin();
        const auto tns_end = opt_dist_tns_it.value()->second.cend();
        for ( ; tns_it != tns_end; ++tns_it )
            optional_tns.emplace(
                std::make_pair(
                    tileidx_t( tns_it->first ),
                    std::make_optional( tns_it )
                )
            );
    }
    else
    {
        for ( const auto& [tile_index, tile_nc] : grid_nc.tiles_node_coord_map_ )
            optional_tns.emplace(
                std::make_pair(
                    tileidx_t( tile_index ),
                    std::optional< Tns_IT >()
                )
            );
    }

    return optional_tns;
}


template < typename CoordT >
std::forward_list< const Tile< CoordT >* >
inline get_optional_masked_sub_tiles(
    const Tile< CoordT >* const& tile,
    const MaskCollection< CoordT >* const& mask_collection
)
{
    std::forward_list< const Tile< CoordT >* > leaf_sub_tiles;

    if ( mask_collection->has_source_mask() )
        leaf_sub_tiles = mask_collection->source_overlapping_leafs( tile );
    else
        tile->insert_leaf_sub_tiles( leaf_sub_tiles );

    return leaf_sub_tiles;
}


template < typename CoordT >
IndexedNodeSequenceJointureMap
get_optional_node_sequences(
    const std::optional< Tns_IT >& opt_tns_it,
    const NodeIdxCoordMap< CoordT >& coord_map
)
{
    if ( opt_tns_it.has_value() )
        return collect_indexed_node_sequence_jointures(
            NodeSequence( opt_tns_it.value()->second ),
            coord_map
        );

    IndexedNodeSequenceJointureMap ns_map;
    for ( const auto& [node_index, node_vec] : coord_map )
        ns_map.emplace(
            std::make_pair(
                nodeidx_t( node_index ),
                NodeSequence( nodeidx_t( node_index ),
                    static_cast< nodeidx_t >( node_vec.size() ) )
            )
        );

    return ns_map;
}


template < typename CoordT >
inline void check_insert_node_coord_pairs(
    std::vector< std::pair< nodeidx_t, CoordT > >& node_coord_pairs,
    const std::optional< Tns_IT >& opt_tns_it,
    const NodeIdxCoordMap< CoordT >& coord_map,
    const MaskCollection< CoordT >* const& mask_collection
)
{
    const auto jointures = get_optional_node_sequences(
        opt_tns_it,
        coord_map
    );

    filter_insert_jointures(
        node_coord_pairs,
        jointures,
        coord_map,
        mask_collection,
        false // inverted_source_target
    );
}


template < typename CoordT >
ConsolidatedNodeCoordMap< CoordT >
slice_tiled_node_maps(
    const DistributedTiledNodeSequenceMap* const& dist_tns,
    const TileGrid< CoordT >& tile_grid,
    const GridNeighborhood& grid_neighborhood,
    const GridNodeCollection< CoordT >& grid_node_col,
    const TAArray< MaskCollection< CoordT > >& mc_array
)
{
    assert(
        tile_grid.has_split_ &&
        grid_neighborhood.has_owners_ &&
        !grid_node_col.tiles_node_coord_map_.empty() &&
        mc_array.is_initialized()
    );

    ConsolidatedNodeCoordMap< CoordT > tiled_coord_map;
    std::optional< DistTns_IT > opt_dist_tns_it;
    if ( dist_tns != nullptr )
    {
        assert( !dist_tns->empty() );
        auto local_tile_seq_map_it = dist_tns->find( grid_neighborhood.local_rank_ );
        if ( local_tile_seq_map_it == dist_tns->end() )
            return tiled_coord_map;

        opt_dist_tns_it.emplace( std::move( local_tile_seq_map_it ) );
    }

#pragma omp parallel default( none )\
shared( tile_grid, grid_node_col, mc_array, tiled_coord_map )\
firstprivate( opt_dist_tns_it )
#pragma omp master
#pragma omp taskgroup
    {
        const auto mask_collection = mc_array.get_local_thread_item().get();
        for ( const auto& [tile_index, opt_tns_it] :
            get_optional_tiled_node_sequences( opt_dist_tns_it, grid_node_col ) )
        {

            const auto tile_nc_it = grid_node_col.tiles_node_coord_map_.find( tile_index );
            assert( tile_nc_it != grid_node_col.tiles_node_coord_map_.end() );
            const auto tile_pos_it = tile_grid.positions_.cbegin() + tile_index;

            const auto overlapping_leafs = get_optional_masked_sub_tiles(
                tile_pos_it->get_tile(),
                mask_collection
            );

            if ( overlapping_leafs.empty() ) continue;

            const auto emplace_tile_it = tiled_coord_map.emplace(
                std::make_pair(
                    tileidx_t( tile_index ),
                    std::unordered_map< tileidx_t,
                    std::vector< std::pair< nodeidx_t, CoordT > > >()
                )
            ).first;

            for ( const auto& leaf_ptr : overlapping_leafs )
            {
                const auto coord_map_it =
                    tile_nc_it->second.sub_tiles_node_coord_map_.find( leaf_ptr->index_ );
                assert( coord_map_it != tile_nc_it->second.sub_tiles_node_coord_map_.end() );

                if ( coord_map_it->second.empty() ) continue;

                const auto emplace_leaf_res = emplace_tile_it->second.emplace(
                    std::make_pair(
                        tileidx_t( coord_map_it->first ),
                        std::vector< std::pair< nodeidx_t, CoordT > >()
                    )
                );
                assert( emplace_leaf_res.second );

#pragma omp task default( none ) shared( mc_array )\
firstprivate( emplace_leaf_res, opt_tns_it, coord_map_it )
                check_insert_node_coord_pairs(
                    emplace_leaf_res.first->second,
                    opt_tns_it,
                    coord_map_it->second,
                    mc_array.get_local_thread_item().get()
                );
            }
        }
    }

    auto tile_it = tiled_coord_map.begin(); // guaranteed to be non empty
    const auto tile_end = tiled_coord_map.end();
    while ( tile_it != tile_end )
    {
        auto leaf_it = tile_it->second.begin(); // guaranteed to be non empty
        const auto leaf_end = tile_it->second.end();
        while ( leaf_it != leaf_end )
        {
            if ( leaf_it->second.empty() )
                leaf_it = tile_it->second.erase( leaf_it );
            else
                ++leaf_it;
        }

        if ( tile_it->second.empty() )
            tile_it = tiled_coord_map.erase( tile_it );
        else
            ++tile_it;
    }

    return tiled_coord_map;
}
}


#endif
