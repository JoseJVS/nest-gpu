/*
 *  api_converters.cpp
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

#include "timer_manager.h"
#include "api_converters.h"

namespace sapi
{
std::vector< std::string >
nested_charray_to_string_vector( const NestedCharArray& nca )
{
    if ( nca.size_ < 1 )
        return {};
    if ( nca.array_ == nullptr )
        throw std::invalid_argument( "Corrupted NestedCharArray" );
    std::vector< std::string > strs( nca.size_ );
    for ( std::size_t idx = 0; idx < nca.size_; ++idx )
        strs[ idx ] = charray_to_string( nca.array_[ idx ] );
    return strs;
}


void copy_to_nested_charray_from_string_vector(
    NestedCharArray& nca,
    const std::vector< std::string >& str_vec,
    GC& gc
)
{
    nca.resize( str_vec.size(), gc );
    std::size_t idx = 0;
    for ( const auto& str : str_vec )
        copy_to_charray_from_string( nca.array_[ idx++ ], str, gc );
}


void copy_to_tns_pair_array_from_dist_tns_map(
    TiledNodeSequencePairArray& tnspa,
    const DistributedTiledNodeSequenceMap& dtns,
    GC& gc
)
{
    tnspa.resize( dtns.size(), gc );

    std::size_t rix = 0;
    for ( const auto& [rank, tiled_node_sequence] : dtns )
    {
        const auto rank_tm_pp = &tnspa.array_[ rix++ ];
        rank_tm_pp->first_ = rank;
        rank_tm_pp->second_.resize( tiled_node_sequence.size(), gc );

        std::size_t tix = 0;
        for ( const auto& [tile_index, node_sequence] : tiled_node_sequence )
        {
            const auto tile_ns_pp = &rank_tm_pp->second_.array_[ tix++ ];
            tile_ns_pp->first_ = tile_index;
            tile_ns_pp->second_.first_ = node_sequence.first;
            tile_ns_pp->second_.second_ = node_sequence.second;
        }
    }
}


void copy_to_conn_pair_array_from_conn_info_map(
    ConnectionInfoPairArray& cpa,
    const std::unordered_map< vp_t, TileConnectionInfo >& conn_info_map,
    GC& gc
)
{
    cpa.resize( conn_info_map.size(), gc );

    std::size_t rix = 0;
    for ( const auto& [rank, tile_ci] : conn_info_map )
    {
        const auto rank_ntm_pp = &cpa.array_[ rix++ ];
        rank_ntm_pp->first_ = rank;
        rank_ntm_pp->second_.resize( tile_ci.partitioned_connection_vectors_.size(), gc );

        std::size_t pix = 0;
        for ( const auto& conn_vec : tile_ci.partitioned_connection_vectors_ )
        {
            const auto partition_ap = &rank_ntm_pp->second_.array_[ pix++ ];
            partition_ap->resize( conn_vec.sizes_, gc );

            for ( count_t conn_idx = 0; conn_idx < conn_vec.sizes_; ++conn_idx )
            {
                const auto ci_struct_p = &partition_ap->array_[ conn_idx ];
                ci_struct_p->source_index_ = conn_vec.connection_sources_[ conn_idx ];
                ci_struct_p->target_index_ = conn_vec.connection_targets_[ conn_idx ];
                ci_struct_p->connection_weight_ = conn_vec.connection_weights_[ conn_idx ];
                ci_struct_p->connection_delay_ = conn_vec.connection_delays_[ conn_idx ];
            }
        }
    }
}


void copy_to_nc_pair_array_from_anycoord_map(
    NodeCoordPairArray& ncpa,
    const TileIdxNodeIdxACM& anycoord_map,
    GC& gc
)
{
    ncpa.resize( anycoord_map.size(), gc );

    std::size_t tix = 0;
    for ( const auto& [tile_index, node_coord_map] : anycoord_map )
    {
        const auto tile_ncm_pp = &ncpa.array_[ tix++ ];
        tile_ncm_pp->first_ = tile_index;
        tile_ncm_pp->second_.resize( node_coord_map.size(), gc );

        std::size_t nix = 0;
        for ( const auto& [node_index, anycoord] : node_coord_map )
        {
            const auto node_coord_pp = &tile_ncm_pp->second_.array_[ nix++ ];
            node_coord_pp->first_ = node_index;
            node_coord_pp->second_.resize( anycoord.size(), gc );

            std::size_t dix = 0;
            for ( const auto& dim : anycoord )
                node_coord_pp->second_.array_[ dix++ ] = dim;
        }
    }
}


void copy_to_nested_nc_pair_array_from_nested_anycoord_map(
    NestedNodeCoordPairArray& nested_ncpa,
    const NestedTileIdxNodeIdxACM& nested_anycoord_map,
    GC& gc
)
{
    nested_ncpa.resize( nested_anycoord_map.size(), gc );

    std::size_t tix = 0;
    for ( const auto& [tile_index, leaf_ncm_map] : nested_anycoord_map )
    {
        const auto tile_leaf_ncm_pp = &nested_ncpa.array_[ tix++ ];
        tile_leaf_ncm_pp->first_ = tile_index;
        copy_to_nc_pair_array_from_anycoord_map(
            tile_leaf_ncm_pp->second_,
            leaf_ncm_map,
            gc
        );
    }
}


void copy_to_gtv_pair_array_from_gtv_map(
    GridTileVerticesPairArray& gtvpa,
    const GridTileVertexMap& grid_vertices,
    GC& gc
)
{
    gtvpa.resize( grid_vertices.size(), gc );

    std::size_t tix = 0;
    for ( const auto& [tile_index, vec_map_pair] : grid_vertices )
    {
        const auto tile_vstp_pp = &gtvpa.array_[ tix++ ];
        tile_vstp_pp->first_ = tile_index;
        copy_to_nested_array_from_nested_collection(
            tile_vstp_pp->second_.first_,
            vec_map_pair.first,
            gc
        );

        tile_vstp_pp->second_.second_.resize( vec_map_pair.second.size(), gc );

        std::size_t lix = 0;
        for ( const auto& [leaf_index, anycoord_vec] : vec_map_pair.second )
        {
            const auto leaf_vertices_pp = &tile_vstp_pp->second_.second_.array_[ lix++ ];
            leaf_vertices_pp->first_ = leaf_index;

            copy_to_nested_array_from_nested_collection(
                leaf_vertices_pp->second_,
                anycoord_vec,
                gc
            );
        }
    }
}


void copy_to_timer_data_pair_array_from_timer_data_map(
    RecordedTimesArrayPair& rtap,
    const RecordedTimes& rt_map,
    GC& gc
)
{
    const auto rank_times_pp = &rtap.first_;
    rank_times_pp->resize( rt_map.rank_times_.size(), gc );

    std::size_t tix = 0;
    for ( const auto& [name, time] : rt_map.rank_times_ )
    {
        const auto timer_data_pp = &rank_times_pp->array_[ tix++ ];
        copy_to_charray_from_string( timer_data_pp->first_, name, gc );
        timer_data_pp->second_ = time;
    }

    const auto thread_times_pp = &rtap.second_;
    thread_times_pp->resize( rt_map.thread_times_.size(), gc );

    tix = 0;
    for ( const auto& [name, times] : rt_map.thread_times_ )
    {
        const auto thread_arr_pp = &thread_times_pp->array_[ tix++ ];
        copy_to_charray_from_string( thread_arr_pp->first_, name, gc );
        copy_to_array_from_collection( thread_arr_pp->second_, times, gc );
    }
}


MaskParameters mpstruct_to_mask_params(
    const MPStruct& mps
)
{
    MaskParameters mp;

    mp.mask_blueprint_name_ = charray_to_string( mps.mask_blueprint_name_ );
    mp.mask_blueprint_params_ = array_to_vector( mps.mask_blueprint_params_ );
    mp.mask_blueprint_offset_ = array_to_vector( mps.mask_blueprint_offset_ );
    mp.source_mask_name_ = charray_to_string( mps.source_mask_name_ );
    mp.source_mask_origin_ = array_to_vector( mps.source_mask_origin_ );
    mp.source_mask_params_ = array_to_vector( mps.source_mask_params_ );
    mp.source_mask_offset_ = array_to_vector( mps.source_mask_offset_ );
    mp.target_mask_name_ = charray_to_string( mps.target_mask_name_ );
    mp.target_mask_origin_ = array_to_vector( mps.target_mask_origin_ );
    mp.target_mask_params_ = array_to_vector( mps.target_mask_params_ );
    mp.target_mask_offset_ = array_to_vector( mps.target_mask_offset_ );

    return mp;
}


ConnectionParameters cpstruct_to_conn_params(
    const CPStruct& cps
)
{
    ConnectionParameters cp;

    cp.edge_wrap_ = cps.edge_wrap_;
    cp.only_neighborhood_ = cps.only_neighborhood_;
    cp.inverted_conn_rule_ = cps.inverted_conn_rule_;
    cp.allow_multiplicity_ = cps.allow_multiplicity_;
    cp.allow_self_connections_ = cps.allow_self_connections_;
    cp.partition_connections_by_source_ = cps.partition_connections_by_source_;
    cp.connection_counts_ = cps.connection_counts_;

    cp.conn_gen_name_ = charray_to_string( cps.conn_gen_name_ );

    cp.weight_df_name_ = charray_to_string( cps.weight_df_name_ );
    cp.weight_df_params_ = array_to_vector( cps.weight_df_params_ );
    cp.weight_ufs_names_ = nested_charray_to_string_vector( cps.weight_ufs_names_ );
    cp.weight_ufs_params_ = nested_array_to_nested_vector( cps.weight_ufs_params_ );

    cp.delay_df_name_ = charray_to_string( cps.delay_df_name_ );
    cp.delay_df_params_ = array_to_vector( cps.delay_df_params_ );
    cp.delay_ufs_names_ = nested_charray_to_string_vector( cps.delay_ufs_names_ );
    cp.delay_ufs_params_ = nested_array_to_nested_vector( cps.delay_ufs_params_ );

    cp.prob_df_name_ = charray_to_string( cps.prob_df_name_ );
    cp.prob_df_params_ = array_to_vector( cps.prob_df_params_ );
    cp.prob_ufs_names_ = nested_charray_to_string_vector( cps.prob_ufs_names_ );
    cp.prob_ufs_params_ = nested_array_to_nested_vector( cps.prob_ufs_params_ );

    return cp;
}
}
