/*
 *  mask_tile_processing.h
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

#ifndef MASK_TILE_PROCESSING_H
#define MASK_TILE_PROCESSING_H

#include <iterator>

#include "node_collection.h"
#include "mask_containers.h"
#include "mask_collection.h"
#include "grid_neighborhood.h"
#include "thread_aligned_array.h"


namespace sapi
{
template < typename CoordT >
bool tile_overlap(
    const Tile< CoordT >* const& driver,
    const Tile< CoordT >* const& pool,
    const MaskCollection< CoordT >* const& mask_collection,
    const bool& pool_is_shifted,
    const bool& inverted_source_target
)
{
    assert( driver != nullptr && pool != nullptr );

    if ( no_overlap( driver, mask_collection, inverted_source_target ) )
        return false;
    if ( !pool_is_shifted && no_overlap( pool, mask_collection, !inverted_source_target ) )
        return false;

    return mask_collection->blueprint_overlap( driver, pool );
}


template < typename CoordT >
void insert_leaf_tiles_within_range(
    std::forward_list< LeafPairInfo< CoordT > >& lpi_list,
    tileidx_t& used_image_indexes,
    const Tile< CoordT >* const& driver,
    const Tile< CoordT >* const& pool,
    const tileidx_t& image_index,
    const std::optional< CoordT >& image_displacement,
    const TAArray< MaskCollection< CoordT > >& mc_array,
    const bool& inverted_connection_rule,
    const bool& inverted_source_target,
    const split_t& splits
)
{
    assert(
        driver->is_split() == pool->is_split() &&
        0 <= splits
    );

    if ( !driver->is_split() )
    {
        // Aggregation is performed on local first
        LeafPairInfo< CoordT > lpi(
            inverted_connection_rule
            ? pool->index_
            : driver->index_,
            inverted_connection_rule
            ? driver->index_
            : pool->index_,
            image_index,
            image_displacement
        );

#pragma omp atomic
        used_image_indexes |= 1 << image_index;

#pragma omp critical
        lpi_list.emplace_front( std::move( lpi ) );
    }
    else
    {
        assert(
            driver->get_sub_tiles().size() < std::numeric_limits< vertidx_t >::max() &&
            pool->get_sub_tiles().size() < std::numeric_limits< vertidx_t >::max()
        );

        const auto d_st_count = static_cast< vertidx_t >( driver->get_sub_tiles().size() );
        const auto d_sub_tiles = driver->get_sub_tiles().data();
        const auto p_st_count = static_cast< vertidx_t >( pool->get_sub_tiles().size() );
        const auto p_sub_tiles = pool->get_sub_tiles().data();
        const auto rem_splits = 0 < splits ? splits - 1 : 0;

#pragma omp taskloop collapse( 2 ) grainsize( 1 ) default( none )\
    shared( lpi_list, used_image_indexes, mc_array )\
    firstprivate( d_st_count, d_sub_tiles, p_st_count, p_sub_tiles,\
        image_index, image_displacement,\
        inverted_connection_rule, inverted_source_target, rem_splits )\
    mergeable final( rem_splits < 2 )
        for ( vertidx_t d_st = 0; d_st < d_st_count; ++d_st )
        {
            for ( vertidx_t p_st = 0; p_st < p_st_count; ++p_st )
            {
                const auto st_driver = d_sub_tiles[ d_st ].get();
                const auto st_pool = p_sub_tiles[ p_st ].get();
                if ( tile_overlap(
                    st_driver, st_pool, mc_array.get_local_thread_item().get(),
                    image_displacement.has_value(), inverted_source_target ) )
                {
                    insert_leaf_tiles_within_range(
                        lpi_list,
                        used_image_indexes,
                        st_driver,
                        st_pool,
                        image_index,
                        image_displacement,
                        mc_array,
                        inverted_connection_rule,
                        inverted_source_target,
                        rem_splits
                    );
                }
            }
        }
    }
}


template < typename CoordT >
void aggregate_leaf_pairs(
    TilePairInfo< CoordT >& tpi,
    const tileidx_t& used_image_indexes,
    const tileidx_t& total_image_indexes
)
{
    if ( tpi.flattened_leaf_pairs_.empty() )
        return;

    std::vector< tileidx_t > index_shifts( total_image_indexes, -1 );

    tileidx_t valid_images = 0;
    for ( tileidx_t index = 0; index < total_image_indexes; ++index )
    {
        if ( 0 < ( used_image_indexes & 1 << index ) )
            index_shifts[ index ] = valid_images++;
    }

    assert( 0 < valid_images );

    auto leaf_pair_move_it = std::make_move_iterator( tpi.flattened_leaf_pairs_.begin() );
    while ( !tpi.flattened_leaf_pairs_.empty() )
    {
        auto [
            source_idx,
            target_idx,
            image_index,
            image_displacement
        ] = *leaf_pair_move_it++;
        tpi.flattened_leaf_pairs_.pop_front();

        const auto shifted_index = index_shifts[ image_index ];
        assert( 0 <= shifted_index );

        const auto source_leaf_it = tpi.aggregated_leaf_pairs_.find( source_idx );
        if ( source_leaf_it == tpi.aggregated_leaf_pairs_.end() )
        {
            std::vector< std::optional< CoordT > > shift_displacements( valid_images );
            shift_displacements[ shifted_index ] = std::move( image_displacement );

            std::unordered_map< tileidx_t,
                std::vector< std::optional< CoordT > > > target_leaf_map;
            target_leaf_map.emplace(
                std::make_pair(
                    tileidx_t( target_idx ),
                    std::move( shift_displacements )
                )
            );

            tpi.aggregated_leaf_pairs_.emplace(
                std::make_pair(
                    tileidx_t( source_idx ),
                    std::move( target_leaf_map )
                )
            );

            continue;
        }

        const auto target_leaf_it = source_leaf_it->second.find( target_idx );
        if ( target_leaf_it == source_leaf_it->second.end() )
        {
            std::vector< std::optional< CoordT > > shift_displacements( valid_images );
            shift_displacements[ shifted_index ] = std::move( image_displacement );

            source_leaf_it->second.emplace(
                std::make_pair(
                    tileidx_t( target_idx ),
                    std::move( shift_displacements )
                )
            );

            continue;
        }

        target_leaf_it->second[ shifted_index ] = std::move( image_displacement );
    }
}


template < typename CoordT >
void tile_pair_overlap(
    TilePairInfo< CoordT >& tpi,
    const TilePosition< CoordT >& driver_tile_pos,
    const TilePosition< CoordT >& pool_tile_pos,
    const TAArray< MaskCollection< CoordT > >& mc_array,
    const bool& edge_wrap,
    const bool& inverted_connection_rule,
    const bool& inverted_source_target,
    const split_t& splits
)
{
    assert( tpi.aggregated_leaf_pairs_.empty() );

    tileidx_t image_index = 0;
    tileidx_t used_image_indexes = 0;
    const auto driver_tile = driver_tile_pos.tile_.get(); // using unique_ptr::get
    const auto pool_tile = pool_tile_pos.tile_.get();
    const auto mask_collection = mc_array.get_local_thread_item().get();

#pragma omp taskgroup
    driver_tile->initialize_sub_tiles( splits );

    if ( edge_wrap )
    {
        assert( !pool_tile_pos.grid_images_.empty() );

#pragma omp taskgroup
        for ( const auto& pool_image : pool_tile_pos.grid_images_ )
        {
            const auto tile_image = pool_image.shifted_tile_
                ? pool_image.shifted_tile_.get()
                : pool_tile;

            if ( !mask_collection->blueprint_overlap(
                driver_tile, tile_image )
                )
                continue;

#pragma omp taskgroup
            tile_image->initialize_sub_tiles( splits );

#pragma omp task default( none ) shared( pool_image, tpi, used_image_indexes, mc_array )\
            firstprivate( image_index, driver_tile, tile_image,\
                inverted_connection_rule, inverted_source_target, splits )
            insert_leaf_tiles_within_range(
                tpi.flattened_leaf_pairs_,
                used_image_indexes,
                driver_tile,
                tile_image,
                image_index,
                pool_image.shift_displacement_,
                mc_array,
                inverted_connection_rule,
                inverted_source_target,
                0 < splits ? splits - 1 : 0
            );

            ++image_index;
        }
    }
    else
    {
        if ( !mask_collection->blueprint_overlap(
            driver_tile, pool_tile )
            )
            return;

#pragma omp taskgroup
        pool_tile->initialize_sub_tiles( splits );

#pragma omp taskgroup
        insert_leaf_tiles_within_range(
            tpi.flattened_leaf_pairs_,
            used_image_indexes,
            driver_tile,
            pool_tile,
            image_index++,
            std::optional< CoordT >(),
            mc_array,
            inverted_connection_rule,
            inverted_source_target,
            0 < splits ? splits - 1 : 0
        );
    }

    if ( 0 < image_index )
#pragma omp task default( none ) shared( tpi ) firstprivate( used_image_indexes, image_index )
        aggregate_leaf_pairs( tpi, used_image_indexes, image_index );
}


template < typename CoordT >
void filter_insert_jointures(
    std::vector< std::pair< nodeidx_t, CoordT > >& node_coord_pairs,
    const IndexedNodeSequenceJointureMap& jointures,
    const NodeIdxCoordMap< CoordT >& coord_map,
    const MaskCollection< CoordT >* const& mask_collection,
    const bool& inverted_source_target
)
{
    assert( node_coord_pairs.empty() );
    if ( jointures.empty() ) return;

    nodeidx_t node_count = 0;
    std::forward_list< std::pair< nodeidx_t, CoordT > > ncp_fl;
    for ( const auto& [original_index, jointure] : jointures )
    {
        const auto nic_it = coord_map.find( original_index );
        assert( nic_it != coord_map.end() );

        // Due to jointure computation the following is guaranteed to be positive or null
        auto coord_vec_it = nic_it->second.cbegin() + ( jointure.first - original_index );
        assert( jointure.second <= std::distance( coord_vec_it, nic_it->second.cend() ) );

        for ( nodeidx_t coord_count = 0; coord_count < jointure.second; ++coord_count )
        {
            if ( no_overlap(
                *coord_vec_it,
                mask_collection,
                inverted_source_target ) )
                continue;

            ++node_count;
            ncp_fl.emplace_front(
                std::make_pair( jointure.first + coord_count, *coord_vec_it++ )
            );
        }
    }

    if ( ncp_fl.empty() ) return;
    assert( 0 < node_count );

    node_coord_pairs.resize( node_count );
    auto ncp_vec_rmove_it = node_coord_pairs.rbegin(); // inverse front insert order of FL
    auto ncp_fl_move_it = std::make_move_iterator( ncp_fl.begin() );
    while ( !ncp_fl.empty() )
    {
        *ncp_vec_rmove_it++ = *ncp_fl_move_it++;
        ncp_fl.pop_front();
    }
}


template < typename CoordT >
inline void check_insert_node_coord_pairs(
    std::vector< std::pair< nodeidx_t, CoordT > >& node_coord_pairs,
    const NodeSequence& node_sequence,
    const NodeIdxCoordMap< CoordT >& coord_map,
    const MaskCollection< CoordT >* const& mask_collection,
    const bool& inverted_source_target
)
{
    const auto jointures = collect_indexed_node_sequence_jointures(
        node_sequence,
        coord_map
    );

    filter_insert_jointures(
        node_coord_pairs,
        jointures,
        coord_map,
        mask_collection,
        inverted_source_target
    );
}


template < typename CoordT >
void aggregate_tile_set_info(
    TileSetInfo< CoordT >& tsi,
    const tileidx_t& source_tile_idx,
    const std::unordered_map< tileidx_t, TilePairInfo< CoordT > >& tpi_map,
    const NodeSequence& node_sequence,
    const TileIdxNodeIdxCoordMap< CoordT >& leaf_coord_map,
    const TAArray< MaskCollection< CoordT > >& mc_array,
    const bool& inverted_source_target
)
{
    assert( !leaf_coord_map.empty() );

    // At this point the given source tile pos
    // was checked against all target tile pos
    // from a given target rank.
    // Hence, tpi_vec is complete and unique
    // to the source/target ranks pair.

    // Keys: leaf sub tile indexes
    std::unordered_map< tileidx_t,
        std::vector< std::pair< nodeidx_t, CoordT > > >
        checked_leaves;

    for ( const auto& [tile_index, tpi] : tpi_map )
        for ( const auto& [leaf_index, target_map] : tpi.aggregated_leaf_pairs_ )
            checked_leaves.insert(
                { leaf_index, {} }
            );

#pragma omp taskgroup
    for ( auto leaf_it = checked_leaves.begin();
        leaf_it != checked_leaves.end();
        ++leaf_it )
    {
        const auto ncm_it = leaf_coord_map.find( leaf_it->first );
        assert( ncm_it != leaf_coord_map.end() );

        if ( ncm_it->second.empty() ) continue;

#pragma omp task default( none ) shared( mc_array )\
firstprivate( leaf_it, node_sequence, ncm_it, inverted_source_target )
        check_insert_node_coord_pairs(
            leaf_it->second,
            node_sequence,
            ncm_it->second,
            mc_array.get_local_thread_item().get(),
            inverted_source_target
        );
    }

    nodeidx_t total_nodes = 0;
    auto check_it = checked_leaves.begin();
    const auto check_end = checked_leaves.end();
    while ( check_it != check_end )
    {
        if ( check_it->second.empty() )
            check_it = checked_leaves.erase( check_it );
        else
            total_nodes += ( *check_it++ ).second.size();
    }

    if ( checked_leaves.empty() ) return;
    assert( 0 < total_nodes );

#pragma omp atomic
    ++tsi.valid_tiles_;
#pragma omp atomic
    tsi.valid_leaves_ += checked_leaves.size();
#pragma omp atomic
    tsi.total_node_count_ += total_nodes;

#pragma omp critical
    tsi.flattened_tile_idx_leaf_nodes_coords_.emplace_front(
        std::make_pair(
            tileidx_t( source_tile_idx ),
            std::move( checked_leaves )
        )
    );
}


template < typename CoordT >
void aggregate_communication_info(
    RankPairInfo< CoordT >& rpi
)
{
    auto tile_map_pair_it = std::make_move_iterator(
        rpi.tile_pairs_set_.flattened_tile_idx_leaf_nodes_coords_.begin()
    );
    while ( !rpi.tile_pairs_set_.flattened_tile_idx_leaf_nodes_coords_.empty() )
    {
        const auto emplace_res = rpi.sender_info_.tile_idx_leaf_nodes_coords_map_.emplace(
            *tile_map_pair_it++
        );
        assert( emplace_res.second );
        rpi.tile_pairs_set_.flattened_tile_idx_leaf_nodes_coords_.pop_front();
    }
}


template < typename CoordT >
void rank_pair_overlap(
    RankPairInfo< CoordT >& rpi,
    const TileIdxNodeSequenceMap& tns_source,
    const TileIdxNodeSequenceMap& tns_target,
    const TileGrid< CoordT >& tile_grid,
    const GridNodeCollection< CoordT >& grid_node_col,
    const TAArray< MaskCollection< CoordT > >& mc_array,
    const bool& edge_wrap,
    const bool& only_neighborhood,
    const bool& inverted_connection_rule,
    const bool& inverted_source_target,
    const bool& prepare_payload
)
{
    assert(
        rpi.tile_pairs_set_.tile_pairs_info_map_.empty() &&
        !tns_target.empty()
    );

    const auto mask_collection = mc_array.get_local_thread_item().get();

#pragma omp taskgroup
    for ( const auto& [source_tile_index, source_node_sequence] : tns_source )
    {
        const auto source_tile_pos = tile_grid.positions_.cbegin() + source_tile_index;

        if ( no_overlap(
            source_tile_pos->get_tile(),
            mask_collection,
            inverted_source_target
        ) ) continue;

        const auto source_emplace_it =
            rpi.tile_pairs_set_.tile_pairs_info_map_.emplace(
                std::make_pair(
                    tileidx_t( source_tile_index ),
                    std::unordered_map< tileidx_t, TilePairInfo< CoordT > >()
                )
        ).first;

#pragma omp taskgroup
        for ( const auto& [target_tile_index, target_node_sequence] : tns_target )
        {
            if ( only_neighborhood && !source_tile_pos->in_neighborhood( target_tile_index, edge_wrap ) )
                continue;

            const auto target_tile_pos = tile_grid.positions_.cbegin() + target_tile_index;

            if ( no_overlap(
                target_tile_pos->get_tile(),
                mask_collection,
                !inverted_source_target
            ) ) continue;

            const auto target_emplace_it =
                source_emplace_it->second.emplace(
                    std::make_pair(
                        tileidx_t( target_tile_index ),
                        TilePairInfo< CoordT >()
                    )
                ).first;

            if ( inverted_connection_rule )
                tile_pair_overlap(
                    target_emplace_it->second,
                    *target_tile_pos,
                    *source_tile_pos,
                    mc_array,
                    edge_wrap,
                    inverted_connection_rule,
                    !inverted_source_target,
                    tile_grid.splits_
                );
            else
                tile_pair_overlap(
                    target_emplace_it->second,
                    *source_tile_pos,
                    *target_tile_pos,
                    mc_array,
                    edge_wrap,
                    inverted_connection_rule,
                    inverted_source_target,
                    tile_grid.splits_
                );
        }

        if ( !source_emplace_it->second.empty() )
        {
            const auto tile_nc_it = grid_node_col.tiles_node_coord_map_.find( source_tile_index );
            assert( tile_nc_it != grid_node_col.tiles_node_coord_map_.end() );

#pragma omp task default( none ) shared( rpi, mc_array )\
firstprivate( tile_nc_it, source_emplace_it, source_node_sequence, inverted_source_target )
            aggregate_tile_set_info(
                rpi.tile_pairs_set_,
                source_emplace_it->first,
                source_emplace_it->second,
                source_node_sequence,
                tile_nc_it->second.sub_tiles_node_coord_map_,
                mc_array,
                inverted_source_target
            );
        }
    }

#pragma omp task default( none ) shared( rpi )\
firstprivate( prepare_payload )
    {
        aggregate_communication_info( rpi );
        if ( prepare_payload )
            prepare_data_payload( rpi.sender_info_, rpi.tile_pairs_set_ );
    }
}


template < typename CoordT >
void distributed_overlap(
    std::unordered_map< vp_t, RankPairInfo< CoordT > >& rpi_map,
    const TileIdxNodeSequenceMap& tns_source,
    const DistributedTiledNodeSequenceMap& dist_tns_target,
    const TileGrid< CoordT >& tile_grid,
    const GridNeighborhood& grid_neighborhood,
    const GridNodeCollection< CoordT >& grid_node_col,
    const TAArray< MaskCollection< CoordT > >& mc_array,
    const bool& edge_wrap,
    const bool& only_neighborhood,
    const bool& inverted_connection_rule,
    const bool& inverted_source_target,
    const bool& force_prepare_payload
)
{
    assert( rpi_map.empty() && !tns_source.empty() );

    for ( const auto& [rank, tns_target] : dist_tns_target )
    {
        const auto is_remote = rank != grid_neighborhood.local_rank_;
        if ( only_neighborhood && is_remote &&
            !grid_neighborhood.in_neighborhood( rank, edge_wrap) )
            continue;

        const auto emplace_it = rpi_map.emplace(
            std::make_pair(
                vp_t( rank ),
                RankPairInfo< CoordT >()
            )
        ).first;

        rank_pair_overlap(
            emplace_it->second,
            tns_source,
            tns_target,
            tile_grid,
            grid_node_col,
            mc_array,
            edge_wrap,
            only_neighborhood,
            inverted_connection_rule,
            inverted_source_target,
            force_prepare_payload || is_remote
        );
    }
}


template < typename CoordT >
std::pair< bool, bool >
compute_distributed_tile_overlap(
    DistributedPairInfo< CoordT >& dpi,
    const DistributedTiledNodeSequenceMap& dist_tns_source,
    const DistributedTiledNodeSequenceMap& dist_tns_target,
    const TileGrid< CoordT >& tile_grid,
    const GridNeighborhood& grid_neighborhood,
    const GridNodeCollection< CoordT >& grid_node_col,
    const TAArray< MaskCollection< CoordT > >& mc_array,
    const bool& edge_wrap,
    const bool& only_neighborhood,
    const bool& inverted_connection_rule,
    const bool& force_prepare_payload = false
)
{
    assert(
        !dist_tns_source.empty() &&
        !dist_tns_target.empty() &&
        tile_grid.has_split_ &&
        grid_neighborhood.has_owners_ &&
        !grid_node_col.tiles_node_coord_map_.empty() &&
        mc_array.is_initialized() &&
        mc_array.get_local_thread_item()->has_blueprint()
    );

    bool is_source = false;
    bool is_target = false;
    const auto source_side = dist_tns_source.find( grid_neighborhood.local_rank_ );
    const auto target_side = dist_tns_target.find( grid_neighborhood.local_rank_ );

#pragma omp taskgroup
    {
        if ( source_side != dist_tns_source.end() )
        {
            distributed_overlap(
                dpi.source_side_info_,
                source_side->second,
                dist_tns_target,
                tile_grid,
                grid_neighborhood,
                grid_node_col,
                mc_array,
                edge_wrap,
                only_neighborhood,
                inverted_connection_rule,
                false, // inverted_source_target
                force_prepare_payload
            );
        }

        if ( target_side != dist_tns_target.end() )
        {
            distributed_overlap(
                dpi.target_side_info_,
                target_side->second,
                dist_tns_source,
                tile_grid,
                grid_neighborhood,
                grid_node_col,
                mc_array,
                edge_wrap,
                only_neighborhood,
                !inverted_connection_rule,
                true, // inverted_source_target
                force_prepare_payload
            );
        }
    }

    is_source = !dpi.source_side_info_.empty();
    is_target = !dpi.target_side_info_.empty();

    return std::make_pair(
        is_source || is_target,
        is_source && is_target
    );
}
}


#endif
