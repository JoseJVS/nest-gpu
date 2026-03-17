/*
 *  api_converters.h
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

#ifndef SAPI_CONVERTERS_H
#define SAPI_CONVERTERS_H

#include <set>
#include <stdexcept>
#include <type_traits>

#include "api_containers.h"
#include "node_containers.h"
#include "spatial_containers.h"
#include "connection_containers.h"


namespace sapi
{
// Forward definition to link with timer_manager.h
struct RecordedTimes;


inline std::string
charray_to_string( const CharArray& ca )
{
    if ( ca.size_ < 1 )
        return "";
    if ( ca.array_ == nullptr ||
        ca.array_[ ca.size_ - 1 ] != '\0' )
        throw std::invalid_argument( "Corrupted CharArray" );
    return std::string( ca.array_ );
}


inline void
copy_to_charray_from_string(
    CharArray& ca,
    const std::string& str,
    GC& gc
)
{
    const auto str_size = str.size();
    ca.resize( str_size + 1, gc );
    if ( str_size > 0 )
        str.copy( ca.array_, str_size );
    ca.array_[ str_size ] = '\0';
}


inline CharArray*
string_to_charray( const std::string& str, GC& gc )
{
    auto ca_ptr = gc.make_collected< CharArray >();
    copy_to_charray_from_string( *ca_ptr, str, gc );
    return ca_ptr;
}


std::vector< std::string >
nested_charray_to_string_vector( const NestedCharArray& );


void copy_to_nested_charray_from_string_vector(
    NestedCharArray&,
    const std::vector< std::string >&,
    GC&
);


inline NestedCharArray*
string_vector_to_nested_charray(
    const std::vector< std::string >& str_vec,
    GC& gc
)
{
    auto nca = gc.make_collected< NestedCharArray >();
    copy_to_nested_charray_from_string_vector( *nca, str_vec, gc );
    return nca;
}


template < typename T,
    std::enable_if_t<
    std::is_arithmetic_v< T >,
    bool
    > = true
>
std::vector< T >
array_to_vector( const ArrayT< T >& ta )
{
    if ( ta.size_ < 1 )
        return {};
    if ( ta.array_ == nullptr )
        throw std::invalid_argument( "Corrupted ArrayT" );
    std::vector< T > vec( ta.size_ );
    for ( std::size_t idx = 0; idx < ta.size_; ++idx )
        vec[ idx ] = ta.array_[ idx ];
    return vec;
}


template < typename T,
    std::enable_if_t<
    std::is_arithmetic_v< T >,
    bool
    > = true
>
std::vector< std::vector< T > >
nested_array_to_nested_vector( const ArrayT< ArrayT< T > >& na )
{
    if ( na.size_ < 1 )
        return {};
    if ( na.array_ == nullptr )
        throw std::invalid_argument( "Corrupted NestedTarray" );
    std::vector< std::vector< T > > nvec( na.size_ );
    for ( std::size_t idx = 0; idx < na.size_; ++idx )
        nvec[ idx ] = array_to_vector( na.array_[ idx ] );
    return nvec;
}


template < typename T,
    std::enable_if_t<
    std::is_arithmetic_v< T >,
    bool
    > = true
>
std::set< T >
array_to_set( const ArrayT< T >& ta )
{
    if ( ta.size_ < 1 )
        return {};
    if ( ta.array_ == nullptr )
        throw std::invalid_argument( "Corrupted ArrayT" );
    std::set< T > set;
    for ( std::size_t idx = 0; idx < ta.size_; ++idx )
        set.insert( ta.array_[ idx ] );
    if ( set.size() != ta.size_ )
        throw std::invalid_argument( "Corrupted set from ArrayT" );
    return set;
}


template < typename T,
    std::enable_if_t<
    std::is_arithmetic_v< T >,
    bool
    > = true
>
std::vector< std::set< T > >
nested_array_to_set_vector( const ArrayT< ArrayT< T > >& na )
{
    if ( na.size_ < 1 )
        return {};
    if ( na.array_ == nullptr )
        throw std::invalid_argument( "Corrupted NestedTarray" );
    std::vector< std::set< T > > setvec( na.size_ );
    for ( std::size_t idx = 0; idx < na.size_; ++idx )
        setvec[ idx ] = array_to_set( na.array_[ idx ] );
    return setvec;
}


template < typename CollectionT,
    std::enable_if_t<
    std::is_arithmetic_v< typename CollectionT::value_type >,
    bool
    > = true
>
void copy_to_array_from_collection(
    ArrayT< typename CollectionT::value_type >& ta,
    const CollectionT& ct,
    GC& gc
)
{
    ta.resize( ct.size(), gc );
    std::size_t t_idx = 0;
    for ( const auto& t : ct )
        ta.array_[ t_idx++ ] = t;
}


template < typename CollectionT,
    std::enable_if_t<
    std::is_arithmetic_v< typename CollectionT::value_type >,
    bool
    > = true
>
inline ArrayT< typename CollectionT::value_type >*
collection_to_array(
    const CollectionT& ct,
    GC& gc
)
{
    auto ta_ptr = gc.make_collected<
        ArrayT< typename CollectionT::value_type >
    >();
    copy_to_array_from_collection( *ta_ptr, ct, gc );
    return ta_ptr;
}


template < typename NestedCollectionT,
    std::enable_if_t<
    std::is_arithmetic_v< typename NestedCollectionT::value_type::value_type >,
    bool
    > = true
>
void copy_to_nested_array_from_nested_collection(
    ArrayT< ArrayT< typename NestedCollectionT::value_type::value_type > >& nta,
    const NestedCollectionT& nct,
    GC& gc
)
{
    nta.resize( nct.size(), gc );
    std::size_t ta_idx = 0;
    for ( const auto& ct : nct )
        copy_to_array_from_collection( nta.array_[ ta_idx++ ], ct, gc );
}


template < typename NestedCollectionT,
    std::enable_if_t<
    std::is_arithmetic_v< typename NestedCollectionT::value_type::value_type >,
    bool
    > = true
>
inline ArrayT< ArrayT< typename NestedCollectionT::value_type::value_type > >*
nested_collection_to_nested_array(
    const NestedCollectionT& nct,
    GC& gc
)
{
    auto nta_ptr = gc.make_collected<
        ArrayT< ArrayT< typename NestedCollectionT::value_type::value_type > >
    >();
    copy_to_nested_array_from_nested_collection( *nta_ptr, nct, gc );
    return nta_ptr;
}


void copy_to_tns_pair_array_from_dist_tns_map(
    TiledNodeSequencePairArray&,
    const DistributedTiledNodeSequenceMap&,
    GC&
);


inline TiledNodeSequencePairArray*
dist_tns_map_to_tns_pair_array(
    const DistributedTiledNodeSequenceMap& dtns,
    GC& gc
)
{
    auto tnspa_ptr = gc.make_collected< TiledNodeSequencePairArray >();
    copy_to_tns_pair_array_from_dist_tns_map( *tnspa_ptr, dtns, gc );
    return tnspa_ptr;
}


void copy_to_conn_pair_array_from_conn_info_map(
    ConnectionInfoPairArray&,
    const std::unordered_map< vp_t, TileConnectionInfo >&,
    GC&
);


inline ConnectionInfoPairArray*
conn_info_map_to_conn_pair_array(
    const std::unordered_map< vp_t, TileConnectionInfo >& conn_info_map,
    GC& gc
)
{
    auto cpa_ptr = gc.make_collected< ConnectionInfoPairArray >();
    copy_to_conn_pair_array_from_conn_info_map( *cpa_ptr, conn_info_map, gc );
    return cpa_ptr;
}


inline RemoteConnectionInfoPair*
rci_to_rcipair(
    const RankConnectionInfo& rci,
    GC& gc
)
{
    const auto rcip_ptr = gc.make_collected< RemoteConnectionInfoPair >();
    copy_to_conn_pair_array_from_conn_info_map(
        rcip_ptr->first_,
        rci.incoming_connections_,
        gc
    );
    copy_to_conn_pair_array_from_conn_info_map(
        rcip_ptr->second_,
        rci.outgoing_connections_,
        gc
    );
    return rcip_ptr;
}


void copy_to_nc_pair_array_from_anycoord_map(
    NodeCoordPairArray&,
    const TileIdxNodeIdxACM&,
    GC&
);


inline NodeCoordPairArray*
anycoord_map_to_nc_pair_array(
    const TileIdxNodeIdxACM& anycoord_map,
    GC& gc
)
{
    auto ncpa_ptr = gc.make_collected< NodeCoordPairArray >();
    copy_to_nc_pair_array_from_anycoord_map(
        *ncpa_ptr,
        anycoord_map,
        gc
    );
    return ncpa_ptr;
}


void copy_to_nested_nc_pair_array_from_nested_anycoord_map(
    NestedNodeCoordPairArray&,
    const NestedTileIdxNodeIdxACM&,
    GC&
);


inline NestedNodeCoordPairArray*
nested_anycoord_map_to_nested_nc_pair_array(
    const NestedTileIdxNodeIdxACM& nested_anycoord_map,
    GC& gc
)
{
    auto nested_ncpa = gc.make_collected< NestedNodeCoordPairArray >();
    copy_to_nested_nc_pair_array_from_nested_anycoord_map(
        *nested_ncpa, nested_anycoord_map, gc
    );
    return nested_ncpa;
}


void copy_to_gtv_pair_array_from_gtv_map(
    GridTileVerticesPairArray&,
    const GridTileVertexMap&,
    GC&
);


inline GridTileVerticesPairArray*
gtv_map_to_gtv_pair_array(
    const GridTileVertexMap& gtv_map,
    GC& gc
)
{
    auto gtvpa_ptr = gc.make_collected< GridTileVerticesPairArray >();
    copy_to_gtv_pair_array_from_gtv_map( *gtvpa_ptr, gtv_map, gc );
    return gtvpa_ptr;
}


void copy_to_timer_data_pair_array_from_timer_data_map(
    RecordedTimesArrayPair&,
    const RecordedTimes&,
    GC&
);


inline RecordedTimesArrayPair*
recorded_times_to_array_pair(
    const RecordedTimes& rt_map,
    GC& gc
)
{
    auto rtap_ptr = gc.make_collected< RecordedTimesArrayPair >();
    copy_to_timer_data_pair_array_from_timer_data_map(
        *rtap_ptr, rt_map, gc
    );
    return rtap_ptr;
}


MaskParameters mpstruct_to_mask_params(
    const MPStruct&
);


ConnectionParameters cpstruct_to_conn_params(
    const CPStruct&
);
}


#endif
