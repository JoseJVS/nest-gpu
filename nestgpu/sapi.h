#ifndef SAPI_H
#define SAPI_H

#include "api_containers.h"


namespace sapi
{
extern "C"
{
    bool init( vp_t argc, char** argv );

    bool reset_api();

    bool free_gc();

    OptionalIndex get_rank();

    OptionalIndex get_num_processes();

    OptionalIndex get_num_threads();

    bool set_num_threads( const vp_t& num_threads );

    OptionalIndex get_rng_seed();

    bool set_rng_seed( const uint32_t& seed );

    CharArray* get_rng_type();

    bool set_rng_type( const CharArray& rng_type );

    bool generate_tile_grid(
        const SpaceTArray& grid_origin,
        const TileIdxArray& grid_dimensions,
        const CharArray& tile_type,
        const SpaceTArray& tile_params,
        const NestedTileIdxArray& rank_tiles_ownership_map,
        const split_t& num_splits,
        const bool& edge_wrap
    );

    OptionalIndex generate_nodes_in_grid(
        const largenodeidx_t& num_nodes,
        const TileIdxArray& tile_set,
        const uint8_t& grid_distribution_mode,
        const uint8_t& tile_distribution_mode
    );

    PairT< OptionalIndex, NestedSpaceTArray* >
        insert_positions_in_grid(
            const NestedSpaceTArray& positions
        );

    OptionalIndex compute_spatial_connections(
        const std::size_t& dist_tns_source_index,
        const std::size_t& dist_tns_target_index,
        const MPStruct& mask_params,
        const CPStruct& conn_params
    );

    NestedNodeCoordPairArray* get_nodes(
        const PairT< bool, std::size_t >& opt_dist_tns_index,
        const MPStruct& mask_params
    );

    TiledNodeSequencePairArray* get_distributed_node_sequences(
        const std::size_t& dist_tns_index
    );

    RCIStruct* get_spatial_connections(
        const std::size_t& conn_idx
    );

    GridTileVerticesPairArray* get_grid_vertices();

    TimerDataPairArray* get_timer_data();
}
}


#endif
