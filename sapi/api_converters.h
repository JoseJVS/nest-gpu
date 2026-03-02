#ifndef SAPI_CONVERTERS_H
#define SAPI_CONVERTERS_H

#include <set>
#include <stdexcept>

#include "api_containers.h"
#include "node_containers.h"
#include "spatial_containers.h"
#include "connection_containers.h"


namespace sapi
{
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


inline std::vector< std::string >
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


inline void copy_to_nested_charray_from_string_vector(
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


template < typename T >
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


template < typename T >
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


template < typename T >
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


template < typename T >
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


template < typename CollectionT >
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


template < typename CollectionT >
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


template < typename NestedCollectionT >
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


template < typename NestedCollectionT >
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


inline void copy_to_tns_pair_array_from_dist_tns_map(
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


inline void copy_to_conn_pair_array_from_conn_info_map(
    ConnectionInfoArray& cpa,
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
        rank_ntm_pp->second_.resize( tile_ci.total_generated_connections_, gc );

        count_t total_conn_idx = 0;
        for ( const auto& conn_vec : tile_ci.source_unique_connection_vectors_ )
        {
            for ( count_t conn_idx = 0; conn_idx < conn_vec.sizes_; ++conn_idx )
            {
                const auto ci_struct_p = &rank_ntm_pp->second_.array_[ total_conn_idx + conn_idx ];
                ci_struct_p->source_index_ = conn_vec.connection_sources_[ conn_idx ];
                ci_struct_p->target_index_ = conn_vec.connection_targets_[ conn_idx ];
                ci_struct_p->connection_weight_ = conn_vec.connection_weights_[ conn_idx ];
                ci_struct_p->connection_delay_ = conn_vec.connection_delays_[ conn_idx ];
            }
            total_conn_idx += conn_vec.sizes_;
        }

        if ( total_conn_idx != tile_ci.total_generated_connections_ )
            throw std::runtime_error( "Corrupted ConnectionInfoArray" );
    }
}


inline ConnectionInfoArray*
conn_info_map_to_conn_pair_array(
    const std::unordered_map< vp_t, TileConnectionInfo >& conn_info_map,
    GC& gc
)
{
    auto cpa_ptr = gc.make_collected< ConnectionInfoArray >();
    copy_to_conn_pair_array_from_conn_info_map( *cpa_ptr, conn_info_map, gc );
    return cpa_ptr;
}


inline void copy_to_nc_pair_array_from_anycoord_map(
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


inline void copy_to_nested_nc_pair_array_from_nested_anycoord_map(
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


inline void copy_to_gtv_pair_array_from_gtv_map(
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


inline void copy_to_timer_data_pair_array_from_timer_data_map(
    TimerDataPairArray& tdpa,
    const std::unordered_map< std::string, double >& td_map,
    GC& gc
)
{
    tdpa.resize( td_map.size(), gc );

    std::size_t tix = 0;
    for ( const auto& [name, time] : td_map )
    {
        const auto timer_data_pp = &tdpa.array_[ tix++ ];
        copy_to_charray_from_string( timer_data_pp->first_, name, gc );
        timer_data_pp->second_ = time;
    }
}


inline TimerDataPairArray*
timer_data_map_to_timer_data_pair_array(
    const std::unordered_map< std::string, double >& td_map,
    GC& gc
)
{
    auto tdpa_ptr = gc.make_collected< TimerDataPairArray >();
    copy_to_timer_data_pair_array_from_timer_data_map(
        *tdpa_ptr, td_map, gc
    );
    return tdpa_ptr;
}


inline RCIStruct*
rci_to_rcistruct(
    const RankConnectionInfo& rci,
    GC& gc
)
{
    const auto rci_ptr = gc.make_collected< RCIStruct >();
    copy_to_conn_pair_array_from_conn_info_map(
        rci_ptr->incoming_connections_,
        rci.incoming_connections_,
        gc
    );
    copy_to_conn_pair_array_from_conn_info_map(
        rci_ptr->outgoing_connections_,
        rci.outgoing_connections_,
        gc
    );
    return rci_ptr;
}


inline MaskParameters
mpstruct_to_mask_params(
    const MPStruct& mps
)
{
    MaskParameters mp;

    mp.mask_blueprint_name_ = charray_to_string( mps.mask_blueprint_name_ );
    mp.mask_blueprint_params_ = array_to_vector( mps.mask_blueprint_params_ );
    mp.mask_blueprint_offset_ = array_to_vector( mps.mask_blueprint_offset_ );
    mp.source_mask_name_ = charray_to_string( mps.source_mask_name_ );
    mp.source_mask_params_ = array_to_vector( mps.source_mask_params_ );
    mp.source_mask_offset_ = array_to_vector( mps.source_mask_offset_ );
    mp.target_mask_name_ = charray_to_string( mps.target_mask_name_ );
    mp.target_mask_params_ = array_to_vector( mps.target_mask_params_ );
    mp.target_mask_offset_ = array_to_vector( mps.target_mask_offset_ );

    return mp;
}


inline ConnectionParameters
cpstruct_to_conn_params(
    const CPStruct& cps
)
{
    ConnectionParameters cp;

    cp.edge_wrap_ = cps.edge_wrap_;
    cp.only_neighborhood_ = cps.only_neighborhood_;
    cp.inverted_conn_rule_ = cps.inverted_conn_rule_;
    cp.allow_multiplicity_ = cps.allow_multiplicity_;
    cp.allow_self_connections_ = cps.allow_self_connections_;
    cp.total_number_connections_ = cps.total_number_connections_;

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


#endif
