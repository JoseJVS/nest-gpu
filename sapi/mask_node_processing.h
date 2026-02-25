#ifndef MASK_NODE_PROCESSING_H
#define MASK_NODE_PROCESSING_H

#include <iterator>

#include "mask_containers.h"
#include "mask_collection.h"
#include "thread_aligned_array.h"


namespace sapi
{
template < typename CoordT >
inline void check_minimal_displacement(
    std::optional< Displacement< CoordT > >&& computed_displacement,
    std::optional< Displacement< CoordT > >& tracked_minimum
)
{
    tracked_minimum = computed_displacement.has_value()
        ? tracked_minimum.has_value()
        ? leq_test( computed_displacement->distance2_, tracked_minimum->distance2_ )
        ? std::move( computed_displacement )
        : tracked_minimum
        : std::move( computed_displacement )
        : tracked_minimum;
}


template < typename CoordT >
void compute_minimal_displacement(
    std::forward_list< NodeDisplacementInfo< CoordT > >& ndi_list,
    const std::pair< nodeidx_t, CoordT >& driver_node,
    const std::pair< nodeidx_t, CoordT >& pool_node,
    const std::vector< std::optional< CoordT > >& shifted_pairings,
    const MaskCollection< CoordT >* const& mask_collection
)
{
    std::optional< Displacement< CoordT > > min_displacement;
    for ( const auto& shifted_pairing : shifted_pairings )
        check_minimal_displacement(
            mask_collection->blueprint_overlap(
                driver_node.second,
                shifted_pairing.has_value()
                ? pool_node.second + shifted_pairing.value()
                : pool_node.second
            ),
            min_displacement
        );

    if ( min_displacement.has_value() )
        ndi_list.emplace_front(
            NodeDisplacementInfo< CoordT >(
                nodeidx_t( driver_node.first ),
                nodeidx_t( pool_node.first ),
                std::move( min_displacement.value() )
            )
        );
}


template < typename CoordT >
void compute_displacement_checks_lists(
    std::forward_list< NodeDisplacementInfo< CoordT > >& ndi_list,
    const std::vector< std::pair< nodeidx_t, CoordT > >& driver_node_list,
    const std::vector< std::pair< nodeidx_t, CoordT > >& pool_node_list,
    const std::vector< std::optional< CoordT > >& shifted_pairings,
    const MaskCollection< CoordT >* const& mask_collection
)
{
    assert(
        ndi_list.empty() &&
        !shifted_pairings.empty() &&
        mask_collection->has_blueprint()
    );

    for ( const auto& driver_node : driver_node_list )
        for ( const auto& pool_node : pool_node_list )
            compute_minimal_displacement(
                ndi_list,
                driver_node,
                pool_node,
                shifted_pairings,
                mask_collection
            );
}


template < typename CoordT >
std::pair< bool,
    std::unordered_map < tileidx_t,
    std::pair< bool,
    std::unordered_map < tileidx_t,
    std::forward_list<
    std::forward_list<
    NodeDisplacementInfo< CoordT > > > > > >
> pivot_aggregation_map(
    std::unordered_map< tileidx_t,
    std::unordered_map< tileidx_t,
    std::unordered_map< tileidx_t,
    std::unordered_map< tileidx_t,
    std::forward_list< NodeDisplacementInfo< CoordT > > > > > >& procedural_aggregation_map,
    const bool& inverted_pool_driver
)
{
    bool valid_tiles = false;
    std::unordered_map< tileidx_t,
        std::pair< bool,
        std::unordered_map < tileidx_t,
        std::forward_list<
        std::forward_list<
        NodeDisplacementInfo< CoordT > > > > > >
        pivoted_procedural_map;

    if ( procedural_aggregation_map.empty() )
        return std::make_pair(
            std::move( valid_tiles ),
            std::move( pivoted_procedural_map )
        );

    // Construct dummy iterator to be properly instantiated according
    // to pivot key
    auto pivoted_tile_map_it = pivoted_procedural_map.end();

    for ( auto& [local_t_index, remote_t_map] : procedural_aggregation_map )
    {
        if ( remote_t_map.empty() ) continue;

        if ( !inverted_pool_driver )
            // No need for find as aggregation order guarantees unique local tiles
            pivoted_tile_map_it = pivoted_procedural_map.emplace(
                    std::make_pair(
                        tileidx_t( local_t_index ),
                        std::make_pair(
                            false,
                            std::unordered_map< tileidx_t,
                            std::forward_list<
                            std::forward_list<
                            NodeDisplacementInfo< CoordT > > > >()
                        )
                    )
            ).first;

        for ( auto& [remote_t_index, local_l_map] : remote_t_map )
        {
            if ( local_l_map.empty() ) continue;

            if ( inverted_pool_driver )
            {
                pivoted_tile_map_it = pivoted_procedural_map.find(
                    remote_t_index
                );

                if ( pivoted_tile_map_it == pivoted_procedural_map.end() )
                    pivoted_tile_map_it = pivoted_procedural_map.emplace(
                            std::make_pair(
                                tileidx_t( remote_t_index ),
                                std::make_pair(
                                    false,
                                    std::unordered_map< tileidx_t,
                                    std::forward_list<
                                    std::forward_list<
                                    NodeDisplacementInfo< CoordT > > > >()
                                )
                            )
                    ).first;
            }

            // At this point pivot key should be defined
            assert( pivoted_tile_map_it != pivoted_procedural_map.end() );

            // Construct dummy iterator to be properly instantiated according
            // to pivot key
            auto pivoted_leaf_map_it = pivoted_tile_map_it->second.second.end();

            for ( auto& [local_l_index, remote_l_map] : local_l_map )
            {
                if ( remote_l_map.empty() ) continue;

                if ( !inverted_pool_driver )
                {
                    pivoted_leaf_map_it = pivoted_tile_map_it->second.second.find(
                        local_l_index
                    );

                    if ( pivoted_leaf_map_it == pivoted_tile_map_it->second.second.end() )
                        pivoted_leaf_map_it = pivoted_tile_map_it->second.second.emplace(
                                std::make_pair(
                                    tileidx_t( local_l_index ),
                                    std::forward_list<
                                    std::forward_list<
                                    NodeDisplacementInfo< CoordT > > >()
                                )
                        ).first;
                }

                for ( auto& [remote_l_index, disp_fl] : remote_l_map )
                {
                    if ( disp_fl.empty() ) continue;

                    if ( inverted_pool_driver )
                    {
                        pivoted_leaf_map_it = pivoted_tile_map_it->second.second.find(
                            remote_l_index
                        );

                        if ( pivoted_leaf_map_it == pivoted_tile_map_it->second.second.end() )
                            pivoted_leaf_map_it = pivoted_tile_map_it->second.second.emplace(
                                    std::make_pair(
                                        tileidx_t( remote_l_index ),
                                        std::forward_list<
                                        std::forward_list<
                                        NodeDisplacementInfo< CoordT > > >()
                                    )
                            ).first;
                    }

                    // At this point pivot key should be defined
                    assert( pivoted_leaf_map_it != pivoted_tile_map_it->second.second.end() );

                    pivoted_leaf_map_it->second.emplace_front(
                        std::move( disp_fl )
                    );
                    pivoted_tile_map_it->second.first = true;
                    valid_tiles = true;

                    disp_fl.clear();
                }

                remote_l_map.clear();
            }

            local_l_map.clear();
        }

        remote_t_map.clear();
    }

    procedural_aggregation_map.clear();

    return std::make_pair(
        std::move( valid_tiles ),
        std::move( pivoted_procedural_map )
    );
}


template < typename CoordT >
void merge_node_displacement_info_lists(
    std::forward_list<
    std::forward_list<
    NodeDisplacementInfo< CoordT > > >& ndi_fl2,
    ConsolidatedNodeDisplacementMap< CoordT >& consolidation_map
)
{
    assert( !ndi_fl2.empty() && consolidation_map.empty() );

    auto ndi_fl_it = std::make_move_iterator( ndi_fl2.begin() );
    while ( !ndi_fl2.empty() )
    {
        auto ndi_fl = *ndi_fl_it++;
        ndi_fl2.pop_front();

        auto ndi_it = std::make_move_iterator( ndi_fl.begin() );
        while ( !ndi_fl.empty() )
        {
            auto [
                driver_index,
                pool_index,
                displacement
            ] = *ndi_it++;
            ndi_fl.pop_front();

            const auto driver_search = consolidation_map.find( driver_index );
            if ( driver_search == consolidation_map.end() )
            {
                std::map< nodeidx_t, Displacement< CoordT > > pool_map;
                pool_map.emplace(
                    std::make_pair(
                        nodeidx_t( pool_index ),
                        std::move( displacement )
                    )
                );

                consolidation_map.emplace(
                    std::make_pair(
                        nodeidx_t( driver_index ),
                        std::move( pool_map )
                    )
                );

                continue;
            }

            const auto emplace_res = driver_search->second.emplace(
                    std::make_pair(
                        nodeidx_t( pool_index ),
                        std::move( displacement )
                    )
            );
            assert( emplace_res.second );
        }
    }
}


template < typename CoordT >
void consolidate_aggregation_map(
    DisplacementsInfo< CoordT >& dci,
    const bool& inverted_pool_driver
)
{
    auto [valid_tiles, pivoted_procedural_map] = pivot_aggregation_map(
        dci.procedural_aggregation_map_,
        inverted_pool_driver
    );

    if ( !valid_tiles ) return;

#pragma omp taskgroup
    for ( auto tile_pivot_it = pivoted_procedural_map.begin();
        tile_pivot_it != pivoted_procedural_map.end();
        ++tile_pivot_it
        )
    {
        if ( !tile_pivot_it->second.first ) continue;

        const auto consolidated_tile_map_it = dci.consolidated_info_map_.emplace(
                std::make_pair(
                    tileidx_t( tile_pivot_it->first ),
                    std::unordered_map< tileidx_t,
                    ConsolidatedNodeDisplacementMap< CoordT > >()
                )
        ).first;

        for ( auto leaf_pivot_it = tile_pivot_it->second.second.begin();
            leaf_pivot_it != tile_pivot_it->second.second.end();
            ++leaf_pivot_it
            )
        {
            if ( leaf_pivot_it->second.empty() ) continue;

            const auto consolidated_leaf_map_it = consolidated_tile_map_it->second.emplace(
                    std::make_pair(
                        tileidx_t( leaf_pivot_it->first ),
                        ConsolidatedNodeDisplacementMap< CoordT >()
                    )
            ).first;

#pragma omp task default( none ) firstprivate( leaf_pivot_it, consolidated_leaf_map_it )
            merge_node_displacement_info_lists(
                leaf_pivot_it->second,
                consolidated_leaf_map_it->second
            );
        }
    }
}


template < typename CoordT >
void compute_displacement_checks_across_tile_pairs(
    RankPairInfo< CoordT >& rpi,
    const TAArray< MaskCollection< CoordT > >& mc_array,
    const bool& inverted_pool_driver
)
{
    assert(
        mc_array.is_initialized() &&
        !rpi.tile_pairs_set_.tile_pairs_info_map_.empty() &&
        rpi.displacement_checks_map_.procedural_aggregation_map_.empty() &&
        rpi.displacement_checks_map_.consolidated_info_map_.empty()
    );

    // In the unlikely case that tiles passed mask filter
    // but no nodes in the tile passed the mask filter
    // (possible either in sender and/or receiver side)
    // then no work to be done
    if (
        rpi.sender_info_.tile_idx_leaf_nodes_coords_map_.empty() ||
        rpi.receiver_info_.tile_idx_leaf_nodes_coords_map_.empty()
        )
    {
        rpi.tile_pairs_set_.tile_pairs_info_map_.clear();
        rpi.sender_info_.tile_idx_leaf_nodes_coords_map_.clear();
        rpi.receiver_info_.tile_idx_leaf_nodes_coords_map_.clear();
        return;
    }

#pragma omp taskgroup
    // Loop over tile pairings aggregated by local tile
    for ( auto local_tile_tpi_map_pair_it = rpi.tile_pairs_set_.tile_pairs_info_map_.cbegin();
        local_tile_tpi_map_pair_it != rpi.tile_pairs_set_.tile_pairs_info_map_.cend();
        ++local_tile_tpi_map_pair_it )
    {
        // Get aggregated local node coordinates
        const auto local_leaf_node_map_it =
            rpi.sender_info_.tile_idx_leaf_nodes_coords_map_.find(
                local_tile_tpi_map_pair_it->first
            );

        // Local tile index not found -> filtered out during masking
        if (
            local_leaf_node_map_it ==
            rpi.sender_info_.tile_idx_leaf_nodes_coords_map_.end() ||
            local_tile_tpi_map_pair_it->second.empty()
            ) continue;

        // Create aggregation map for local tile
        const auto local_tile_aggregation_it =
            rpi.displacement_checks_map_.procedural_aggregation_map_.emplace(
                std::make_pair(
                    tileidx_t( local_tile_tpi_map_pair_it->first ),
                    std::unordered_map< tileidx_t,
                    std::unordered_map< tileidx_t,
                    std::unordered_map< tileidx_t,
                    std::forward_list<
                    NodeDisplacementInfo< CoordT > > > > >()
            )
        ).first;

        // Loop over remote tile pairings
        for ( auto remote_tile_tpi_map_pair_it = local_tile_tpi_map_pair_it->second.cbegin();
            remote_tile_tpi_map_pair_it != local_tile_tpi_map_pair_it->second.cend();
            ++remote_tile_tpi_map_pair_it )
        {
            // Get aggregated remote node coordinates
            const auto remote_leaf_node_map_it =
                rpi.receiver_info_.tile_idx_leaf_nodes_coords_map_.find(
                    remote_tile_tpi_map_pair_it->first
                );

            // Remote tile not found -> filtered out during masking
            if (
                remote_leaf_node_map_it ==
                rpi.receiver_info_.tile_idx_leaf_nodes_coords_map_.end() ||
                remote_tile_tpi_map_pair_it->second.aggregated_leaf_pairs_.empty()
                ) continue;

            // For each local tile create aggregation map over remote tiles
            const auto remote_tile_aggregation_it =
                local_tile_aggregation_it->second.emplace(
                    std::make_pair(
                        tileidx_t( remote_tile_tpi_map_pair_it->first ),
                        std::unordered_map< tileidx_t,
                        std::unordered_map< tileidx_t,
                        std::forward_list<
                        NodeDisplacementInfo< CoordT > > > >()
                    )
                ).first;

            // Loop over paired leafs aggregated by source leaf
            for ( auto source_leaf_target_map_pair_it =
                remote_tile_tpi_map_pair_it->second.aggregated_leaf_pairs_.cbegin();
                source_leaf_target_map_pair_it !=
                remote_tile_tpi_map_pair_it->second.aggregated_leaf_pairs_.cend();
                ++source_leaf_target_map_pair_it )
            {
                // Leaf pairing procedure guarantees that map entries
                // are only generated for matching pairs
                assert( !source_leaf_target_map_pair_it->second.empty() );

                // Get node coordinates in local leaf
                const auto local_node_vec_it =
                    local_leaf_node_map_it->second.find(
                        source_leaf_target_map_pair_it->first
                    );

                // Local leaf not found -> filtered out during masking
                if (
                    local_node_vec_it ==
                    local_leaf_node_map_it->second.end()
                    ) continue;

                // If the local leaf is found then its vector cannot be empty
                assert( !local_node_vec_it->second.empty() );

                // For each local/remote tile pairs create local leaf aggregation map
                const auto local_leaf_aggregation_it =
                    remote_tile_aggregation_it->second.emplace(
                        std::make_pair(
                            tileidx_t( source_leaf_target_map_pair_it->first ),
                            std::unordered_map< tileidx_t,
                            std::forward_list<
                            NodeDisplacementInfo< CoordT > > >()
                        )
                    ).first;

                // Loop over target leaf pairs
                for ( auto target_shift_set_pair_it = source_leaf_target_map_pair_it->second.cbegin();
                    target_shift_set_pair_it != source_leaf_target_map_pair_it->second.cend();
                    ++target_shift_set_pair_it )
                {
                    // Leaf pairing procedure guarantees that map entries
                    // are only generated for matching pairs
                    assert( !target_shift_set_pair_it->second.empty() );

                    // Get node coordinates in remote leaf
                    const auto remote_node_vec_it =
                        remote_leaf_node_map_it->second.find(
                            target_shift_set_pair_it->first
                        );

                    // Remote leaf not found -> filtered out during masking
                    if (
                        remote_node_vec_it ==
                        remote_leaf_node_map_it->second.end()
                        ) continue;

                    // If the remote leaf is found then its vector cannot be empty
                    assert( !remote_node_vec_it->second.empty() );

                    // For each leaf aggregation map create a node aggregation map 
                    const auto node_aggregation_it =
                        local_leaf_aggregation_it->second.emplace(
                            std::make_pair(
                                tileidx_t( target_shift_set_pair_it->first ),
                                std::forward_list< NodeDisplacementInfo< CoordT > >()
                            )
                        ).first;

#pragma omp task default( none )\
shared( mc_array )\
firstprivate( local_node_vec_it, remote_node_vec_it,\
    node_aggregation_it, target_shift_set_pair_it, inverted_pool_driver )
                    compute_displacement_checks_lists(
                        node_aggregation_it->second,
                        inverted_pool_driver ? remote_node_vec_it->second : local_node_vec_it->second,
                        inverted_pool_driver ? local_node_vec_it->second : remote_node_vec_it->second,
                        target_shift_set_pair_it->second,
                        mc_array.get_local_thread_item().get()
                    );
                }
            }
        }
    }

    rpi.tile_pairs_set_.tile_pairs_info_map_.clear();
    rpi.sender_info_.tile_idx_leaf_nodes_coords_map_.clear();
    rpi.receiver_info_.tile_idx_leaf_nodes_coords_map_.clear();
    consolidate_aggregation_map( rpi.displacement_checks_map_, inverted_pool_driver );
}
}


#endif
