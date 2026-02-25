#ifndef SPATIAL_MANAGER_H
#define SPATIAL_MANAGER_H

#include "nf_creators.h"
#include "cg_creators.h"
#include "gf_creators.h"
#include "node_slicing.h"
#include "mask_creators.h"
#include "node_creation.h"
#include "node_insertion.h"
#include "grid_generation.h"
#include "spatial_containers.h"
#include "spatial_communication.h"


namespace sapi
{
class BaseSpatialManager
{
public:
    BaseSpatialManager() = default;
    BaseSpatialManager( const BaseSpatialManager& ) = delete;
    BaseSpatialManager( BaseSpatialManager&& ) = delete;
    virtual ~BaseSpatialManager() = default;

    BaseSpatialManager( const vp_t& local_rank, const vp_t& num_processes )
        : grid_neighborhood_( local_rank, num_processes )
        , random_manager_( local_rank, num_processes )
    {
    };

    void set_rng_seed( const uint32_t& seed );

    void set_rng_type( const std::string& name );

    virtual void update_num_threads() = 0;

    virtual void initialize_tile_grid(
        const std::vector< space_t >& grid_origin,
        const std::vector< tileidx_t >& grid_dimensions,
        const std::string& tile_type,
        const std::vector< space_t >& tile_params,
        std::vector< std::set< tileidx_t > >&& rank_tiles_ownership_map,
        const split_t& num_splits,
        const bool& edge_wrap
    ) = 0;

    virtual NodeCountVector
        distribute_nodes_in_grid(
            const largenodeidx_t& num_nodes,
            const std::optional< std::set< tileidx_t > >& tile_set,
            const uint8_t& mode
        ) = 0;

    virtual NodeCountVector
        insert_positions_in_grid(
            std::vector< std::vector< space_t > >& anycoord_vec
        ) = 0;

    virtual std::size_t
        generate_nodes_in_tiles(
            const RankNodeSequenceMap& node_sequence_map,
            const uint8_t& mode
        ) = 0;

    virtual std::size_t
        insert_positions_in_tiles(
            const RankNodeSequenceMap& node_sequence_map
        ) = 0;

    virtual std::pair< std::size_t, RankConnectionInfo* >
        compute_spatial_connections(
            const std::size_t& dist_tns_source_index,
            const std::size_t& dist_tns_target_index,
            const MaskParameters& mask_params,
            const ConnectionParameters& conn_params
        ) = 0;

    virtual NestedTileIdxNodeIdxACM
        get_nodes(
            const std::optional< std::size_t >& dist_tns_index,
            const MaskParameters& mask_params
        ) = 0;

    virtual GridTileVertexMap
        get_grid_vertices() = 0;

    const DistributedTiledNodeSequenceMap&
        get_distributed_node_sequence(
            const std::size_t& dist_tns_index
        ) const;

    const RankConnectionInfo&
        get_connection_map(
            const std::size_t& connection_map_index
        ) const;

    TimerData get_timer_data() const;

protected:
    GridNeighborhood grid_neighborhood_;
    TimerRegister timer_register_;
    RandomManager random_manager_;
    std::unordered_map< std::size_t,
        DistributedTiledNodeSequenceMap >
        cached_distributed_tiled_node_sequences_;
    std::unordered_map< std::size_t,
        RankConnectionInfo > cached_connection_maps_;
};


inline void BaseSpatialManager::set_rng_seed( const uint32_t& seed )
{
    random_manager_.set_rng_seed( seed );
}


inline void BaseSpatialManager::set_rng_type( const std::string& name )
{
    random_manager_.set_rng_type( name );
}


inline const DistributedTiledNodeSequenceMap&
BaseSpatialManager::get_distributed_node_sequence(
    const std::size_t& dist_tns_index
) const
{
    const auto search = cached_distributed_tiled_node_sequences_.find(
        dist_tns_index
    );
    if ( search == cached_distributed_tiled_node_sequences_.end() )
        throw std::invalid_argument( "Incorrect cached distributed tiled node sequence index" );

    return search->second;
}


inline const RankConnectionInfo&
BaseSpatialManager::get_connection_map(
    const std::size_t& connection_map_index
) const
{
    const auto search = cached_connection_maps_.find(
        connection_map_index
    );
    if ( search == cached_connection_maps_.end() )
        throw std::invalid_argument( "Incorrect cached connection map index" );

    return search->second;
}


inline TimerData BaseSpatialManager::get_timer_data() const
{
    return timer_register_.to_map();
}


template < typename CoordT >
class SpatialManager : public BaseSpatialManager
{
public:
    SpatialManager();
    SpatialManager( const SpatialManager& ) = delete;
    SpatialManager( SpatialManager&& ) = delete;

    SpatialManager( const vp_t& local_rank, const vp_t& num_processes );

    void update_num_threads() override;

    void initialize_tile_grid(
        const std::vector< space_t >& grid_origin,
        const std::vector< tileidx_t >& grid_dimensions,
        const std::string& tile_type,
        const std::vector< space_t >& tile_params,
        std::vector< std::set< tileidx_t > >&& rank_tiles_ownership_map,
        const split_t& num_splits,
        const bool& edge_wrap
    ) override;

    NodeCountVector
        distribute_nodes_in_grid(
            const largenodeidx_t& num_nodes,
            const std::optional< std::set< tileidx_t > >& tile_set,
            const uint8_t& mode
        ) override;

    NodeCountVector
        insert_positions_in_grid(
            std::vector< std::vector< space_t > >& anycoord_vec
        ) override;

    std::size_t
        generate_nodes_in_tiles(
            const RankNodeSequenceMap& node_sequence_map,
            const uint8_t& mode
        ) override;

    std::size_t
        insert_positions_in_tiles(
            const RankNodeSequenceMap& node_sequence_map
        ) override;

    std::pair< std::size_t, RankConnectionInfo* >
        compute_spatial_connections(
            const std::size_t& dist_tns_source_index,
            const std::size_t& dist_tns_target_index,
            const MaskParameters& mask_params,
            const ConnectionParameters& conn_params
        ) override;

    NestedTileIdxNodeIdxACM
        get_nodes(
            const std::optional< std::size_t >& dist_tns_index,
            const MaskParameters& mask_params
        ) override;

    GridTileVertexMap
        get_grid_vertices() override;

protected:
    bool initialized_grid_ = false;
    TileGrid< CoordT > tile_grid_;
    GridNodeCollection< CoordT > grid_node_col_;
    CreatorRegistry< UnaryFunctor >  uf_registry_;
    CreatorRegistry< Mask< CoordT > > mk_registry_;
    CreatorRegistry< DisplacementFunctor< CoordT > >  df_registry_;
    CreatorRegistry< ConnectionGenerator< CoordT > >  cg_registry_;
    CreatorRegistry< BaseCachedTileCreator< CoordT > > ctc_registry_;
    CreatorRegistry< GridTargetPositionShifts< CoordT > > gsc_registry_;
    CreatorRegistry< BaseShiftedOriginCreator< CoordT > > soc_registry_;
    TAArray< GFCollection< CoordT > > gc_array_;
    TAArray< MaskCollection< CoordT > > mc_array_;
    TAArray< ConnectionGenerator< CoordT > > cg_array_;

    // Temporaries for distribution/insertion of nodes in tiles
    // after getting rank node sequences
    std::forward_list< TileIdxNodeCountPairListVector >
        temp_node_generation_data_;
    std::forward_list< std::pair<
        TileIdxNodeCountPairListVector, TiledCoordMap< CoordT >
    > >
        temp_node_insertion_data_;

    void _initialize_registers();

    void _initialize_mask_parameters(
        const MaskParameters& mask_params
    );

    void _initialize_connection_parameters(
        const ConnectionParameters& conn_params
    );
};


template < typename CoordT >
SpatialManager< CoordT >::SpatialManager()
    : BaseSpatialManager()
{
    _initialize_registers();
}


template < typename CoordT >
SpatialManager< CoordT >::SpatialManager( const vp_t& local_rank, const vp_t& num_processes )
    : BaseSpatialManager( local_rank, num_processes )
{
    _initialize_registers();
}


template < typename CoordT >
inline void SpatialManager< CoordT >::_initialize_registers()
{
    initialize_mk_registry( mk_registry_ );
    initialize_uf_registry( uf_registry_ );
    initialize_df_registry( df_registry_ );
    initialize_cg_registry( cg_registry_ );
    initialize_gsc_registry( gsc_registry_ );
    initialize_soc_registry( soc_registry_ );
    initialize_ctc_registry( ctc_registry_ );
}


template < typename CoordT >
inline void SpatialManager< CoordT >::update_num_threads()
{
    gc_array_.prepare();
    mc_array_.prepare();
    cg_array_.prepare();
    random_manager_.initialize();
}


template < typename CoordT >
void SpatialManager< CoordT >::initialize_tile_grid(
    const std::vector< space_t >& grid_origin,
    const std::vector< tileidx_t >& grid_dimensions,
    const std::string& tile_type,
    const std::vector< space_t >& tile_params,
    std::vector< std::set< tileidx_t > >&& rank_tiles_ownership_map,
    const split_t& num_splits,
    const bool& edge_wrap
)
{
    if ( initialized_grid_ )
        throw std::runtime_error( "Cannot generate tile grid more than once" );

    const auto ggt = timer_register_.get_register_timer( "grid_generation_time" );
    ggt->start();

    const auto gc = GFCollection< CoordT >(
        grid_origin,
        grid_dimensions,
        tile_type,
        tile_params,
        gsc_registry_,
        soc_registry_,
        ctc_registry_
    );

    if ( !gc.check_dimensions( edge_wrap ) )
        throw std::invalid_argument(
            "Grid cannot be instantiated with the dimension | rotation | edge wrapping combination"
        );

    gc_array_.clone( gc );

    tile_grid_ = generate_tile_grid(
        gc.get_grid_dimensions(),
        gc_array_,
        edge_wrap
    );

    ggt->stop();

    const auto got = timer_register_.get_register_timer( "grid_ownership_time" );
    got->start();

    grid_neighborhood_.set_tile_ownership(
        std::move( rank_tiles_ownership_map ),
        tile_grid_
    );

    got->stop();

    const auto gst = timer_register_.get_register_timer( "grid_splitting_time" );
    gst->start();

    tile_grid_.split_owned_tiles(
        num_splits,
        *grid_neighborhood_.local_owned_tiles_
    );

    gst->stop();

    const auto gnt = timer_register_.get_register_timer( "grid_node_collections_time" );
    gnt->start();

    grid_node_col_.initialize_map(
        *grid_neighborhood_.local_owned_tiles_,
        tile_grid_
    );

    gnt->stop();

    const auto ct = timer_register_.get_register_timer( "cleanup_time" );
    ct->start();

    gc_array_.clear();

    ct->stop();

    initialized_grid_ = true;
}


template < typename CoordT >
NodeCountVector
SpatialManager< CoordT >::distribute_nodes_in_grid(
    const largenodeidx_t& num_nodes,
    const std::optional< std::set< tileidx_t > >& tile_set,
    const uint8_t& mode
)
{
    const auto ndt = timer_register_.get_register_timer( "node_distribution_time" );
    ndt->start();

    if ( !initialized_grid_ )
        throw std::runtime_error( "Cannot distribute nodes without generating a tile grid" );

    auto [
        node_counts_per_rank,
        tiled_node_counts_per_rank
    ] = distribute_node_counts_in_grid(
        num_nodes,
        tile_set,
        tile_grid_,
        grid_neighborhood_,
        random_manager_,
        mode
    );

    temp_node_generation_data_.emplace_front(
        std::move( tiled_node_counts_per_rank )
    );

    ndt->stop();

    return node_counts_per_rank;
}


template < typename CoordT >
NodeCountVector
SpatialManager< CoordT >::insert_positions_in_grid(
    std::vector< std::vector< space_t > >& anycoord_vec
)
{
    const auto nit = timer_register_.get_register_timer( "node_insertion_time" );
    nit->start();

    if ( !initialized_grid_ )
        throw std::runtime_error( "Cannot distribute nodes without generating a tile grid" );

    std::list < CoordT > coord_list;
    for ( auto& anycoord : anycoord_vec )
    {
        if ( anycoord.size() != static_cast< std::size_t >( CoordT::D ) )
            throw std::invalid_argument( "Incorrect node position dimensions" );

        coord_list.emplace_back( CoordT::copy_from_vec( anycoord.begin() ) );
        anycoord.clear();
    }
    anycoord_vec.clear();

    auto [
        node_counts_per_rank,
        tiled_node_counts_per_rank,
        tiled_coord_map
    ] = insert_node_positions_in_grid(
        coord_list,
        tile_grid_,
        grid_neighborhood_,
        random_manager_
    );

    temp_node_insertion_data_.emplace_front(
        std::make_pair(
            std::move( tiled_node_counts_per_rank ),
            std::move( tiled_coord_map )
        )
    );

    if ( !coord_list.empty() )
    {
        anycoord_vec.reserve( coord_list.size() );
        auto coord_move_it = std::make_move_iterator( coord_list.begin() );
        while ( !coord_list.empty() )
        {
            auto coord = *coord_move_it++;
            coord_list.pop_front();

            std::vector< space_t > temp( CoordT::D );
            coord.copy_to_vec( temp.begin() );
            anycoord_vec.emplace_back( std::move( temp ) );
        }
    }

    nit->stop();

    return node_counts_per_rank;
}


template < typename CoordT >
std::size_t
SpatialManager< CoordT >::generate_nodes_in_tiles(
    const RankNodeSequenceMap& node_sequence_map,
    const uint8_t& mode
)
{
    const auto nct = timer_register_.get_register_timer( "node_consolidation_time" );
    nct->start();

    if ( temp_node_generation_data_.empty() )
        throw std::invalid_argument( "Incorrect node generation data cache" );

    const auto index = cached_distributed_tiled_node_sequences_.size();
    const auto [emplace, success] = cached_distributed_tiled_node_sequences_.emplace(
        std::make_pair(
            std::size_t( index ),
            consolidate_node_sequences_per_tile_per_rank(
                node_sequence_map,
                temp_node_generation_data_.front()
            )
        )
    );
    if ( !success )
        throw std::runtime_error( "Corrupted distributed tiled node sequences cache" );

    temp_node_generation_data_.pop_front();

    nct->stop();

    const auto ngt = timer_register_.get_register_timer( "node_generation_time" );
    ngt->start();

    if (
        const auto local_tns_it = emplace->second.find(
            grid_neighborhood_.local_rank_
        );
        local_tns_it != emplace->second.end()
        )
    {
        distribute_node_counts_in_tiles(
            grid_node_col_,
            local_tns_it->second,
            tile_grid_,
            random_manager_,
            mode
        );
        random_manager_.update_rank_paired_seed(
            grid_neighborhood_.local_rank_
        );
    }

    ngt->stop();

    return index;
}


template < typename CoordT >
std::size_t
SpatialManager< CoordT >::insert_positions_in_tiles(
    const RankNodeSequenceMap& node_sequence_map
)
{
    const auto nct = timer_register_.get_register_timer( "node_consolidation_time" );
    nct->start();

    if ( temp_node_insertion_data_.empty() )
        throw std::invalid_argument( "Incorrect node insertion data cache" );

    const auto temp_it = temp_node_insertion_data_.begin();

    const auto index = cached_distributed_tiled_node_sequences_.size();
    const auto [emplace, success] = cached_distributed_tiled_node_sequences_.emplace(
        std::make_pair(
            std::size_t( index ),
            consolidate_node_sequences_per_tile_per_rank(
                node_sequence_map,
                temp_it->first
            )
        )
    );
    if ( !success )
        throw std::runtime_error( "Corrupted distributed tiled node sequences cache" );

    nct->stop();

    const auto ngt = timer_register_.get_register_timer( "node_insertion_time" );
    ngt->start();

    if (
        const auto local_tns_it = emplace->second.find(
            grid_neighborhood_.local_rank_
        );
        local_tns_it != emplace->second.end()
        )
        insert_node_positions_in_tiles(
            std::move( temp_it->second ),
            grid_node_col_,
            local_tns_it->second,
            tile_grid_
        );
    else if ( !temp_it->second.empty() )
        throw std::runtime_error( "Corrupted tiled coords map cache" );

    temp_node_insertion_data_.pop_front();

    ngt->stop();

    return index;
}


template < typename CoordT >
void SpatialManager< CoordT >::_initialize_mask_parameters(
    const MaskParameters& mask_params
)
{
    const auto mpt = timer_register_.get_register_timer( "mask_param_time" );
    mpt->start();

    const auto mask_collection = MaskCollection< CoordT >(
        mask_params.mask_blueprint_name_,
        mask_params.mask_blueprint_params_,
        mask_params.source_mask_name_,
        mask_params.source_mask_params_,
        mask_params.target_mask_name_,
        mask_params.target_mask_params_,
        mk_registry_
    );

    mc_array_.clone( mask_collection );

    mpt->stop();
}


template < typename CoordT >
void SpatialManager< CoordT >::_initialize_connection_parameters(
    const ConnectionParameters& conn_params
)
{
    const auto cpt = timer_register_.get_register_timer( "conn_param_time" );
    cpt->start();

    CFCollection< CoordT > cfc;

    cfc.weight_functor_ = ConnectionFunctor< CoordT >(
        conn_params.weight_df_name_,
        conn_params.weight_df_params_,
        df_registry_,
        conn_params.weight_ufs_names_,
        conn_params.weight_ufs_params_,
        uf_registry_
    );

    cfc.delay_functor_ = ConnectionFunctor< CoordT >(
        conn_params.delay_df_name_,
        conn_params.delay_df_params_,
        df_registry_,
        conn_params.delay_ufs_names_,
        conn_params.delay_ufs_params_,
        uf_registry_
    );

    cfc.probability_functor_ = ConnectionFunctor< CoordT >(
        conn_params.prob_df_name_,
        conn_params.prob_df_params_,
        df_registry_,
        conn_params.prob_ufs_names_,
        conn_params.prob_ufs_params_,
        uf_registry_
    );

    const auto cg = cg_registry_.get_creator( conn_params.conn_gen_name_ )->create();
    cg->set_total_num_connections( conn_params.total_number_connections_ );
    cg->set_cf_collection( std::move( cfc ) );

    cg_array_.clone( cg );

    cpt->stop();
}


template < typename CoordT >
std::pair< std::size_t, RankConnectionInfo* >
SpatialManager< CoordT >::compute_spatial_connections(
    const std::size_t& dist_tns_source_index,
    const std::size_t& dist_tns_target_index,
    const MaskParameters& mask_params,
    const ConnectionParameters& conn_params
)
{
    const auto sct = timer_register_.get_register_timer( "spatial_conn_time" );
    sct->start();

    const auto dist_tns_source_it = cached_distributed_tiled_node_sequences_.find(
        dist_tns_source_index
    );
    const auto dist_tns_target_it = cached_distributed_tiled_node_sequences_.find(
        dist_tns_target_index
    );
    if ( dist_tns_source_it == cached_distributed_tiled_node_sequences_.end() ||
        dist_tns_target_it == cached_distributed_tiled_node_sequences_.end() )
        throw std::invalid_argument( "Incorrect distributed tiled node sequences index given for spatial connection" );

    _initialize_mask_parameters( mask_params );
    _initialize_connection_parameters( conn_params );

    auto connection_map = compute_distributed_spatial_connections(
        dist_tns_source_it->second,
        dist_tns_target_it->second,
        tile_grid_,
        grid_neighborhood_,
        grid_node_col_,
        mc_array_,
        cg_array_,
        random_manager_,
        timer_register_,
        conn_params.edge_wrap_,
        conn_params.only_neighborhood_,
        conn_params.inverted_conn_rule_,
        conn_params.allow_multiplicity_,
        conn_params.allow_self_connections_
    );

    const auto index = cached_connection_maps_.size();
    const auto [emplace, success] = cached_connection_maps_.emplace(
        std::make_pair(
            std::size_t( index ),
            std::move( connection_map )
        )
    );
    if ( !success )
        throw std::runtime_error( "Corrupted connection maps cache" );

    sct->stop();

    const auto ct = timer_register_.get_register_timer( "cleanup_time" );
    ct->start();

    for ( const auto& rci_ptr : {
        &emplace->second.incoming_connections_,
        &emplace->second.outgoing_connections_
        } )
        for ( const auto& rank_tci_pair : *rci_ptr )
            random_manager_.update_rank_paired_seed( rank_tci_pair.first );

    mc_array_.clear();
    cg_array_.clear();

    ct->stop();

    return std::make_pair(
        index,
        &emplace->second
    );
}


template < typename CoordT >
NestedTileIdxNodeIdxACM
SpatialManager< CoordT >::get_nodes(
    const std::optional< std::size_t >& dist_tns_index,
    const MaskParameters& mask_params
)
{
    const auto sst = timer_register_.get_register_timer( "spatial_slicing_time" );
    sst->start();

    DistributedTiledNodeSequenceMap* dist_tns = nullptr;

    if ( dist_tns_index.has_value() )
    {
        const auto dist_tns_it = cached_distributed_tiled_node_sequences_.find(
            dist_tns_index.value()
        );
        if ( dist_tns_it == cached_distributed_tiled_node_sequences_.end() )
            throw std::invalid_argument(
                "Incorrect distributed tiled node sequences index given for spatial slicing"
            );
        dist_tns = &dist_tns_it->second;
    }

    _initialize_mask_parameters( mask_params );

    ConsolidatedNodeCoordMap< CoordT > slice = slice_tiled_node_maps(
        dist_tns,
        tile_grid_,
        grid_neighborhood_,
        grid_node_col_,
        mc_array_
    );

    sst->stop();

    const auto sct = timer_register_.get_register_timer( "slice_conversion_time" );
    sct->start();

    bool correct = true;
    NestedTileIdxNodeIdxACM res;
    for ( auto& [tile_index, tile_map] : slice )
    {
        const auto [t_emplace, t_success] = res.emplace(
            std::make_pair(
                tileidx_t( tile_index ),
                TileIdxNodeIdxACM()
            )
        );
        correct &= t_success;
        if ( !correct ) break;

        for ( auto& [leaf_index, leaf_map] : tile_map )
        {
            const auto [l_emplace, l_success] = t_emplace->second.emplace(
                std::make_pair(
                    tileidx_t( leaf_index ),
                    NodeIdxAnyCoordMap()
                )
            );
            correct &= l_success;
            if ( !correct ) break;

            // Squeeze over leafs
            for ( const auto& [node_index, coord] : leaf_map )
            {
                std::vector< space_t > any_coord( CoordT::D );
                coord.copy_to_vec( any_coord.begin() );
                correct &= l_emplace->second.emplace(
                    std::make_pair(
                        nodeidx_t( node_index ),
                        std::move( any_coord )
                    )
                ).second;

                if ( !correct ) break;
            }

            if ( !correct ) break;
            leaf_map.clear();
        }

        if ( !correct ) break;
        tile_map.clear();
    }
    if ( !correct )
        throw std::runtime_error( "Corrupted spatial slice" );

    slice.clear();

    sct->stop();

    const auto ct = timer_register_.get_register_timer( "cleanup_time" );
    ct->start();

    mc_array_.clear();

    ct->stop();

    return res;
}


template < typename CoordT >
GridTileVertexMap
SpatialManager< CoordT >::get_grid_vertices()
{
    const auto gvt = timer_register_.get_register_timer( "grid_vertex_time" );
    gvt->start();

    if ( !initialized_grid_ )
        throw std::runtime_error( "Cannot get grid vertices without generating a tile grid" );

    bool correct = true;
    tileidx_t position = 0;
    GridTileVertexMap grid_vertices;
    for ( const auto& tile_pos : tile_grid_.positions_ )
    {
        const auto [tp_emplace, tp_success] = grid_vertices.emplace(
            std::make_pair(
                position++,
                std::make_pair(
                    tile_pos.get_tile()->
                    export_vertices_to_nested_vec(),
                    std::unordered_map< tileidx_t,
                    std::vector< std::vector< space_t > > >()
                )
            )
        );
        correct &= tp_success;
        if ( !correct ) break;

        // Tile node collections contain a pointer to each locally owned leaf tiles
        // if tile position is not in tile node collection then its vertices are not copied
        const auto tnc_search = grid_node_col_.tiles_node_coord_map_.find( tp_emplace->first );
        if ( tnc_search == grid_node_col_.tiles_node_coord_map_.end() ) continue;

        for ( const auto& st_ptr : tnc_search->second.sub_tiles_vector_ )
        {
            const auto [st_emplace, st_success] = tp_emplace->second.second.emplace(
                std::make_pair(
                    tileidx_t( st_ptr->index_ ),
                    st_ptr->export_vertices_to_nested_vec()
                )
            );
            correct &= st_success;
            if ( !correct ) break;
        }

        if ( !correct ) break;
    }
    if ( !correct )
        throw std::runtime_error( "Corrupted grid tile vertices" );

    gvt->stop();

    return grid_vertices;
}
}


#endif
