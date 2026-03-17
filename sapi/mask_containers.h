/*
 *  mask_containers.h
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

#ifndef MASK_CONTAINERS_H
#define MASK_CONTAINERS_H

#include <map>
#include <optional>
#include <stdexcept>

#include "node_containers.h"


namespace sapi
{
// Forward definition to link with coordinates.h
template < typename CoordT >
struct Displacement;

// Forward definition to link with tile.h
template < typename CoordT >
class Tile;


template < typename CoordT >
using LeafPairInfo = std::tuple<
    tileidx_t, // source leaf index
    tileidx_t, // target leaf index
    tileidx_t, // unwrapped image index
    std::optional< CoordT > // unwrapping displacement
>;


template < typename CoordT >
struct TilePairInfo
{
    std::forward_list< LeafPairInfo< CoordT > >
        flattened_leaf_pairs_;

    // Local leaf index
    //  -> Remote leaf index
    //      -> shifted pairings
    std::unordered_map< tileidx_t,
        std::unordered_map< tileidx_t,
        std::vector< std::optional< CoordT > > > >
        aggregated_leaf_pairs_;

    TilePairInfo() = default;
    TilePairInfo( const TilePairInfo& ) = delete;
    TilePairInfo( TilePairInfo&& ) = default;
    ~TilePairInfo() = default;

    bool operator==( const TilePairInfo& tpi ) const
    {
        return aggregated_leaf_pairs_ == tpi.aggregated_leaf_pairs_;
    }
};


template< typename CoordT >
struct TileSetInfo
{
    tileidx_t valid_tiles_ = 0;
    tileidx_t valid_leaves_ = 0;
    nodeidx_t total_node_count_ = 0;

    std::forward_list<
        std::pair< tileidx_t,
        std::unordered_map< tileidx_t,
        std::vector< std::pair< nodeidx_t, CoordT > > > > >
        flattened_tile_idx_leaf_nodes_coords_;

    // Stores the tile to tile pair comparison combinations between
    // locally owned tiles and tiles from another rank
    // Local tile index
    //      remote tile index
    //          tile pair info
    std::unordered_map< tileidx_t,
        std::unordered_map< tileidx_t,
        TilePairInfo< CoordT > > >
        tile_pairs_info_map_;

    TileSetInfo() = default;
    TileSetInfo( const TileSetInfo& ) = delete;
    TileSetInfo( TileSetInfo&& ) = default;
    ~TileSetInfo() = default;

    bool operator==( const TileSetInfo& tsi ) const
    {
        return tile_pairs_info_map_ == tsi.tile_pairs_info_map_;
    }
};


// This map is used to store copies of node coordinates after filtering tiles with masks
// Tile index
//  -> sub tile index
//      -> list of pairs of
//          -> node index (from node sequence) and
//          -> coordinate (filtered by map)
template < typename CoordT >
using ConsolidatedNodeCoordMap =
std::unordered_map< tileidx_t,
    std::unordered_map< tileidx_t,
    std::vector< std::pair< nodeidx_t, CoordT > > > >;


template < typename CoordT >
struct CommunicationInfo
{
    // From source rank
    // This map is then copied onto data_payload_
    // to be sent via MPI to receiving rank
    ConsolidatedNodeCoordMap< CoordT >
        tile_idx_leaf_nodes_coords_map_;

    // Payload is composed by:
    //
    // Header:
    //  header length: 1 element
    //  content length: 1 element
    //  number of tiles: 1 element
    //  for each tile:
    //      index of tile: valid tiles count
    //      number of sub tiles: valid tiles count
    //      for each sub tiles
    //          index of sub tile: valid sub tiles count
    //          number of nodes: valid sub tiles count
    //          for each node in sub tile
    //              node index: total nodes
    //
    // Content:
    //  for the coordinate of each node:
    //      each dimension as a double: total sequences count * number of dimensions
    //
    // Payload on sender side must be manually cleared once the target has received the data
    std::vector< int64_t > data_payload_;

    CommunicationInfo() = default;
    CommunicationInfo( const CommunicationInfo& ) = delete;
    CommunicationInfo( CommunicationInfo&& ) = default;
    ~CommunicationInfo() = default;

    bool operator==( const CommunicationInfo& ci ) const
    {
        return tile_idx_leaf_nodes_coords_map_ ==
            ci.tile_idx_leaf_nodes_coords_map_;
    }

    CommunicationInfo& operator=( CommunicationInfo&& ci )
    {
        tile_idx_leaf_nodes_coords_map_ =
            std::move( ci.tile_idx_leaf_nodes_coords_map_ );
        data_payload_ = std::move( ci.data_payload_ );
        return *this;
    }
};


template < typename CoordT >
using NodeDisplacementInfo = std::tuple<
    nodeidx_t,
    nodeidx_t,
    Displacement< CoordT >
>;


// Need ordered node to node displacement checks for
// RNG reproducibility
template < typename CoordT >
using ConsolidatedNodeDisplacementMap =
std::map< nodeidx_t,
    std::map< nodeidx_t,
    Displacement< CoordT > > >;


template < typename CoordT >
struct DisplacementsInfo
{
    // Local tile idx
    //      remote tile idx
    //          local leaf
    //              remote leaf
    //                  local node idx
    //                      partial remote node displacement check list
    std::unordered_map< tileidx_t,
        std::unordered_map< tileidx_t,
        std::unordered_map< tileidx_t,
        std::unordered_map< tileidx_t,
        std::forward_list<
        NodeDisplacementInfo< CoordT > > > > > > procedural_aggregation_map_;


    // Driver tile idx
    //      Driver leaf idx
    //          ( ordered ) driver node idx
    //              ( ordered ) pool node idx
    //                  displacement check list
    std::unordered_map< tileidx_t,
        std::unordered_map< tileidx_t,
        ConsolidatedNodeDisplacementMap< CoordT > > > consolidated_info_map_;

    DisplacementsInfo() = default;
    DisplacementsInfo( const DisplacementsInfo& ) = delete;
    DisplacementsInfo( DisplacementsInfo&& ) = default;
    ~DisplacementsInfo() = default;

    bool operator==( const DisplacementsInfo& dci ) const
    {
        return consolidated_info_map_ == dci.consolidated_info_map_;
    }
};


template < typename CoordT >
struct RankPairInfo
{
    TileSetInfo< CoordT > tile_pairs_set_;
    CommunicationInfo< CoordT > sender_info_;
    CommunicationInfo< CoordT > receiver_info_;
    DisplacementsInfo< CoordT > displacement_checks_map_;

    RankPairInfo() = default;
    RankPairInfo( const RankPairInfo& ) = delete;
    RankPairInfo( RankPairInfo&& ) = default;
    ~RankPairInfo() = default;

    bool operator==( const RankPairInfo& rpi ) const
    {
        return tile_pairs_set_ == rpi.tile_pairs_set_ &&
            sender_info_ == rpi.sender_info_ &&
            receiver_info_ == rpi.receiver_info_ &&
            displacement_checks_map_ == rpi.displacement_checks_map_;
    }
};


template < typename CoordT >
struct DistributedPairInfo
{
    std::unordered_map< vp_t, RankPairInfo< CoordT > > source_side_info_;
    std::unordered_map< vp_t, RankPairInfo< CoordT > > target_side_info_;

    DistributedPairInfo() = default;
    DistributedPairInfo( const DistributedPairInfo& ) = delete;
    DistributedPairInfo( DistributedPairInfo&& ) = default;
    ~DistributedPairInfo() = default;

    bool operator==( const DistributedPairInfo& dci ) const
    {
        return source_side_info_ == dci.source_side_info_
            && target_side_info_ == dci.target_side_info_;
    }
};


template < typename CoordT >
void prepare_data_payload(
    CommunicationInfo< CoordT >& si,
    const TileSetInfo< CoordT >& tsi
)
{
    // Compute total data length
    constexpr const vp_t three = 3;
    constexpr const vp_t two = 2;
    constexpr const vp_t dims = CoordT::D;
    const vp_t total_header_size = three + two * (
        static_cast< vp_t >( tsi.valid_tiles_ )
        + static_cast< vp_t >( tsi.valid_leaves_ )
        ) + static_cast< vp_t >( tsi.total_node_count_ );
    const vp_t total_content_size = static_cast< vp_t >( tsi.total_node_count_ ) * dims;
    const vp_t total_payload_size = total_header_size + total_content_size;

    // Check overflow
    if ( !(
        3 <= total_header_size &&
        0 <= total_content_size &&
        3 <= total_payload_size
        ) )
        throw std::runtime_error( "Could not compute correct total payload size." );

    assert( ( 3 < total_payload_size ) != si.tile_idx_leaf_nodes_coords_map_.empty() );

    // Prepare payload buffer
    si.data_payload_.resize( total_payload_size );

    // Prepare writing iterators
    auto info_writing_pos = si.data_payload_.begin();
    const auto info_writing_end = si.data_payload_.begin() + total_header_size;
    auto content_writing_pos = info_writing_end;
    const auto content_writing_end = si.data_payload_.end();

    // Write fixed header metadata
    *info_writing_pos++ = total_header_size;
    *info_writing_pos++ = total_content_size;
    *info_writing_pos++ = tsi.valid_tiles_;

    for ( const auto& [tile_idx, leaf_ncm] : si.tile_idx_leaf_nodes_coords_map_ )
    {
        assert( !leaf_ncm.empty() );

        // Write tile index and total number of sub tiles in tile
        *info_writing_pos++ = tile_idx;
        *info_writing_pos++ = leaf_ncm.size();

        for ( const auto& [leaf_idx, node_coord_pairs] : leaf_ncm )
        {
            assert( !node_coord_pairs.empty() );

            // Write sub tile index and total number of nodes in sub tile
            *info_writing_pos++ = leaf_idx;
            const auto num_nodes = static_cast< nodeidx_t >( node_coord_pairs.size() );
            *info_writing_pos++ = num_nodes;

            assert(
                num_nodes <= std::distance( info_writing_pos, info_writing_end ) &&
                dims * num_nodes <= std::distance( content_writing_pos, content_writing_end )
            );

            for ( const auto& [node_idx, coord] : node_coord_pairs )
            {
                // Write node index and coord
                *info_writing_pos++ = node_idx;
                coord.bit_copy_to_vec( content_writing_pos );
            }
        }
    }

    assert(
        info_writing_pos == info_writing_end &&
        content_writing_pos == content_writing_end
    );
}


template < typename CoordT >
void reconstruct_received_info( CommunicationInfo< CoordT >& ri )
{
    const auto payload_size = ri.data_payload_.size();
    assert( 3 <= payload_size );

    // If only three elements are received it means
    // the payload content is empty and only
    // minimum header was received
    if ( 3 == payload_size )
    {
        ri.data_payload_.clear();
        return;
    }

    // Prepare reading iterators
    auto info_reading_pos = ri.data_payload_.begin();
    const auto recv_header_length = static_cast< vp_t >( *info_reading_pos++ );
    const auto recv_content_length = static_cast< vp_t >( *info_reading_pos++ );
    const auto recv_payload_length = recv_header_length + recv_content_length;
    assert(
        8 <= recv_header_length && // Minimum 1 tile 1 subtile 1 node -> 8 header items
        0 < recv_content_length &&
        8 < recv_payload_length &&
        recv_content_length % CoordT::D == 0 &&
        payload_size == static_cast< std::size_t >( recv_payload_length )
    );

    const auto info_reading_end = ri.data_payload_.begin() + recv_header_length;
    auto content_reading_pos = info_reading_end;
    const auto content_reading_end = ri.data_payload_.end();

    // Read total number of tiles
    auto recv_num_tiles = static_cast< tileidx_t >( *info_reading_pos++ );
    assert( 0 < recv_num_tiles );
    for ( ; recv_num_tiles > 0; --recv_num_tiles )
    {
        // Read tile index and total number of sub tiles
        const auto recv_tile_idx = static_cast< tileidx_t >( *info_reading_pos++ );
        auto recv_num_sub_tiles = static_cast< tileidx_t >( *info_reading_pos++ );
        assert( 0 < recv_num_sub_tiles );

        std::unordered_map< tileidx_t, std::vector< std::pair< nodeidx_t, CoordT > > >
            leaf_nodes_coords_map;
        for ( ; recv_num_sub_tiles > 0; --recv_num_sub_tiles )
        {
            // Read sub tile index and total number of nodes
            const auto recv_st_idx = static_cast< tileidx_t >( *info_reading_pos++ );
            auto recv_num_nodes = static_cast< nodeidx_t >( *info_reading_pos++ );
            assert( 0 < recv_num_nodes &&
                recv_num_nodes <= std::distance( info_reading_pos, info_reading_end ) &&
                CoordT::D * recv_num_nodes <= std::distance( content_reading_pos, content_reading_end )
            );

            std::vector< std::pair< nodeidx_t, CoordT > > node_coord_pairs( recv_num_nodes );
            auto ncp_it = node_coord_pairs.begin();
            for ( ; recv_num_nodes > 0; --recv_num_nodes )
                // Read node idx and coord data
                *ncp_it++ = std::make_pair(
                    static_cast< nodeidx_t >( *info_reading_pos++ ),
                    CoordT::bit_copy_from_vec( content_reading_pos )
                );

            const auto emplace_res = leaf_nodes_coords_map.emplace(
                std::make_pair(
                    tileidx_t( recv_st_idx ),
                    std::move( node_coord_pairs )
                )
            );
            assert( emplace_res.second );
        }

        const auto emplace_res = ri.tile_idx_leaf_nodes_coords_map_.emplace(
            std::make_pair(
                tileidx_t( recv_tile_idx ),
                std::move( leaf_nodes_coords_map )
            )
        );
        assert( emplace_res.second );
    }

    assert( info_reading_pos == info_reading_end && content_reading_pos == content_reading_end );

    // Automatically clear after successful reconstruction
    ri.data_payload_.clear();
}
}


#endif
