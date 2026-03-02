/*
 *  node_containers.h
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

#ifndef NODE_CONTAINERS_H
#define NODE_CONTAINERS_H

#include <list>
#include <vector>
#include <utility>
#include <algorithm> // Min/Max
#include <forward_list>
#include <unordered_map>
#include <cassert>

#include "sapi_config.h"


namespace sapi
{
// First index in sequence and sequence length
typedef std::pair< nodeidx_t, nodeidx_t > NodeSequence;

// Map of tile index to NodeSequence contained in tile
typedef std::unordered_map< tileidx_t, NodeSequence > TileIdxNodeSequenceMap;

// Map of MPI ranks to NodeSequence instantiated in rank
typedef std::unordered_map< vp_t, NodeSequence > RankNodeSequenceMap;

// Map of MPI ranks to tile index node sequence maps.
// Given a global node creation procedure, a global node sequence is generated,
// this sequence is randomly distributed across the tiles in a grid,
// each tile then holds a sub sequence of the global node sequence .
// As each tile is owned by a single rank,
// tile index node sequence can then be aggregated by MPI rank.
typedef std::unordered_map< vp_t, TileIdxNodeSequenceMap > DistributedTiledNodeSequenceMap;

// Vector containing node counts.
// Given a number of processes or number of tiles,
// the vector holds the number of nodes in each rank or tile index,
// the index of the node count is equal to either the rank or tile index.
typedef std::vector< nodeidx_t > NodeCountVector;

// List of individual mappings of tile index to corresponding node count.
// As tiles owned by an individual rank may not have contiguous indexes,
// node counts need to be paired with the corresponding tile index,
// during node creation the list is iterated to instantiate nodes given their count.
// List is preferred over a map as the iteration process goes over all indexes,
// and no random access is necessary.
typedef std::forward_list< std::pair< tileidx_t, nodeidx_t > > TileIdxNodeCountPairList;

// After instantiating node counts for each tile during global node sequence assignment,
// each rank will hold a list of node counts per owned tile,
// the index of each list corresponds to the rank of the owner.
typedef std::vector< TileIdxNodeCountPairList > TileIdxNodeCountPairListVector;

// Map of tile index to sized forward list of coordinates.
// This map is procedurally generated when manually inserting coordinates
// into the grid using bounding box tree filtering approach.
template < typename CoordT >
using TiledCoordMap = std::unordered_map< tileidx_t, std::list< CoordT > >;

// Map of node index to coordinate.
// A map is instantiated for each tile position in a tile grid.
// Map contains the first node id of a node sequence and
// a vector containing all the coords of the node sequence.
// Length of the vector is equal to the size of the node sequence.
template < typename CoordT >
using NodeIdxCoordMap = std::unordered_map< nodeidx_t, std::vector< CoordT > >;

// When filtering nodes in a node index coord map node sequence jointures are computed
// for each entry in the map, then the original key of the node index coord map is used
// as a key for the joint node sequence found.
typedef std::unordered_map< nodeidx_t, NodeSequence > IndexedNodeSequenceJointureMap;

// This container maps leaf sub tile indexes to node index coord maps.
// In a grid, for each tile, each leaf sub tile owns a node index coord map.
// When generating node coordinates a leaf sub tile is chosen and sequences of node
// coordinates are inserted into the map.
template < typename CoordT >
using TileIdxNodeIdxCoordMap = std::unordered_map< tileidx_t, NodeIdxCoordMap< CoordT > >;


constexpr NodeSequence shift_sequence(
    const NodeSequence& origin,
    const NodeSequence& target,
    const nodeidx_t& shift
)
{
    return { origin.first, std::min< nodeidx_t >(
        std::max< nodeidx_t >( target.second - shift, 0 ),
        origin.second ) };
}


constexpr NodeSequence join_sequences( const NodeSequence& left, const NodeSequence& right )
{
    return ( right.first < left.first )
        ? shift_sequence( left, right, left.first - right.first )
        : shift_sequence( right, left, right.first - left.first );
}


template < typename CoordT >
IndexedNodeSequenceJointureMap
collect_indexed_node_sequence_jointures(
    const NodeSequence& node_sequence,
    const NodeIdxCoordMap< CoordT >& node_index_coord_map
)
{
    assert( 0 <= node_sequence.first && 0 < node_sequence.second );

    // Collect all node sequence jointures
    IndexedNodeSequenceJointureMap jointures;

    // Loop over all node coord sequences in leaf tile
    for ( const auto& [node_index, coord_vec] : node_index_coord_map )
    {
        NodeSequence jointure = join_sequences(
            {
                node_index,
                static_cast< nodeidx_t >( coord_vec.size() )
            },
            node_sequence
        );

        if ( jointure.second < 1 ) continue;

        // Insert into jointure map
        jointures.emplace(
            std::make_pair(
                nodeidx_t( node_index ),
                std::move( jointure )
            )
        );
    }

    return jointures;
}
}


#endif
