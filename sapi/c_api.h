#ifndef C_API_H
#define C_API_H

#include "api_containers.h"
#include "spatial_manager.h"


namespace sapi
{
class CAPI
{
public:
    CAPI() = default;
    CAPI( const CAPI& ) = delete;
    CAPI( CAPI&& ) = default;
    ~CAPI() = default;

    bool is_initialized() const
    {
        return 0 <= local_rank_ && 0 < num_processes_;
    }

    bool manager_initialized() const
    {
        return spatial_manager_ != nullptr;
    }

    void reset();
    void free_gc();

    vp_t get_rank() const;
    void set_rank( const vp_t& rank );

    vp_t get_num_processes() const;
    void set_num_processes( const vp_t& num_processes );

    vp_t get_num_threads() const;
    void set_num_threads( const vp_t& num_threads );

    uint32_t get_rng_seed() const;
    void set_rng_seed( const uint32_t& seed );

    CharArray* get_rng_type();
    void set_rng_type( const CharArray& rng_type );

    void generate_tile_grid(
        const SpaceTArray& grid_origin,
        const TileIdxArray& grid_dimensions,
        const CharArray& tile_type,
        const SpaceTArray& tile_params,
        const NestedTileIdxArray& rank_tiles_ownership_map,
        const split_t& num_splits,
        const bool& edge_wrap
    );

    NodeCountVector
        generate_nodes_in_grid(
            const largenodeidx_t& num_nodes,
            const TileIdxArray& tile_set,
            const uint8_t& distribution_mode
        );

    std::size_t generate_nodes_in_tiles(
        const RankNodeSequenceMap& node_sequence_map,
        const uint8_t& distribution_mode
    );

    std::pair< NodeCountVector, NestedSpaceTArray* >
        insert_positions_in_grid(
            const NestedSpaceTArray& anycoord_array
        );

    std::size_t insert_positions_in_tiles(
        const RankNodeSequenceMap& node_sequence_map
    );

    std::pair< std::size_t, RankConnectionInfo* >
        compute_spatial_connections(
        const std::size_t& dist_tns_source_index,
        const std::size_t& dist_tns_target_index,
        const MPStruct& mask_params,
        const CPStruct& conn_params
    );

    NestedNodeCoordPairArray* get_nodes(
        const OptionalIndex& opt_dist_tns_index,
        const MPStruct& mask_params
    );

    TiledNodeSequencePairArray*
        get_distributed_node_sequences(
        const std::size_t& dist_tns_index
    );

    RCIStruct* get_spatial_connections(
        const std::size_t& conn_map_idx
    );

    GridTileVerticesPairArray* get_grid_vertices();

    TimerDataPairArray* get_timer_data();

private:
    // Needs to be cached here
    // as random manager is initialized
    // in spatial manager
    vp_t local_rank_ = -1;
    vp_t num_processes_ = -1;
    uint32_t seed_ = DEFAULT_BASE_SEED_;
    std::string rng_type_ = DEFAULT_RNG_TYPE_;

    GC gc_;
    std::unique_ptr< BaseSpatialManager > spatial_manager_;
};
}


#endif
