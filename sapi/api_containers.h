#ifndef SAPI_CONTAINERS_H
#define SAPI_CONTAINERS_H

#include "sapi_config.h"
#include "garbage_collector.h"


namespace sapi
{
template < typename L, typename R >
struct PairT
{
    L first_;
    R second_;
};


template < typename T0, typename T1, typename T2 >
struct TripletT
{
    T0 first_;
    T1 second_;
    T2 third_;
};


template < typename T >
struct ArrayT
{
    std::size_t size_ = 0;
    T* array_ = nullptr;

    void resize(
        const std::size_t& size,
        GC& garbage_collector
    )
    {
        size_ = size;
        array_ = garbage_collector.make_collected< T >( size );
    }
};


template < typename K, typename V >
using PairArrayT = ArrayT< PairT< K, V > >;


extern "C"
{
    typedef PairT< bool, std::size_t > OptionalIndex;
    typedef TripletT< std::size_t, nodeidx_t, nodeidx_t > SpatialNodeSequence;
    typedef ArrayT< char > CharArray;
    typedef ArrayT< space_t > SpaceTArray;
    typedef ArrayT< tileidx_t > TileIdxArray;

    typedef ArrayT< CharArray > NestedCharArray;
    typedef ArrayT< SpaceTArray > NestedSpaceTArray;
    typedef ArrayT< TileIdxArray > NestedTileIdxArray;

    typedef PairArrayT< vp_t,
        PairArrayT< tileidx_t,
        PairT< nodeidx_t,
        nodeidx_t > > > TiledNodeSequencePairArray;
    typedef PairArrayT< tileidx_t,
        PairArrayT< nodeidx_t,
        ArrayT< space_t > > > NodeCoordPairArray;
    typedef PairArrayT< tileidx_t,
        NodeCoordPairArray > NestedNodeCoordPairArray;
    typedef PairArrayT < tileidx_t,
        PairT< NestedSpaceTArray,
        PairArrayT< tileidx_t, NestedSpaceTArray > > > GridTileVerticesPairArray;
    typedef PairArrayT< CharArray, double > TimerDataPairArray;

    struct CIStruct
    {
        conn_index_t source_index_;
        conn_index_t target_index_;
        conn_param_t connection_weight_;
        conn_param_t connection_delay_;
    };

    typedef PairArrayT< vp_t,
        ArrayT< CIStruct > > ConnectionInfoArray;

    struct RCIStruct
    {
        ConnectionInfoArray incoming_connections_;
        ConnectionInfoArray outgoing_connections_;
    };

    struct MPStruct
    {
        // At least one required
        CharArray mask_blueprint_name_;
        SpaceTArray mask_blueprint_params_;
        SpaceTArray mask_blueprint_offset_;
        CharArray source_mask_name_;
        SpaceTArray source_mask_params_;
        SpaceTArray source_mask_offset_;
        CharArray target_mask_name_;
        SpaceTArray target_mask_params_;
        SpaceTArray target_mask_offset_;
    };

    struct CPStruct
    {
        // Control parameters
        bool edge_wrap_ = false;
        bool only_neighborhood_ = false;
        bool inverted_conn_rule_ = false;
        bool allow_multiplicity_ = false;
        bool allow_self_connections_ = false;
        mult_t total_number_connections_ = 0;

        // Connection generation
        CharArray conn_gen_name_;

        // Weight computation
        CharArray weight_df_name_;
        SpaceTArray weight_df_params_;
        NestedCharArray weight_ufs_names_;
        NestedSpaceTArray weight_ufs_params_;

        // Delay computation
        CharArray delay_df_name_;
        SpaceTArray delay_df_params_;
        NestedCharArray delay_ufs_names_;
        NestedSpaceTArray delay_ufs_params_;

        // Probability drawing
        CharArray prob_df_name_;
        SpaceTArray prob_df_params_;
        NestedCharArray prob_ufs_names_;
        NestedSpaceTArray prob_ufs_params_;
    };
}
}


#endif
