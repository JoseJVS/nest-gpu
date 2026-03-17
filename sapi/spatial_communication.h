/*
 *  spatial_communication.h
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

#ifndef SPATIAL_COMMUNICATION_H
#define SPATIAL_COMMUNICATION_H

#include "vp_interface.h"
#include "timer_manager.h"
#include "mask_tile_processing.h"
#include "mask_node_processing.h"
#include "connection_generation.h"


namespace sapi
{
template < typename CoordT >
void communicate_connect_distributed_pair_data(
    RankConnectionInfo& rank_connection_map,
    DistributedPairInfo< CoordT >& distributed_pair_data,
    const TAArray< MaskCollection< CoordT > >& mc_array,
    const TAArray< ConnectionGenerator >& cg_array,
    const RandomManager& random_manager,
    const TimerManager& timer_manager,
    const vp_t& local_rank,
    const bool& inverted_connection_rule,
    const bool& allow_multiplicity,
    const bool& partition_connections_by_source,
    const bool& has_local_targets
);


template < typename CoordT >
inline void compute_target_connections(
    RankPairInfo< CoordT >& rpi,
    TileConnectionInfo& tci,
    const vp_t& target_rank,
    const TAArray< MaskCollection< CoordT > >& mc_array,
    const TAArray< ConnectionGenerator >& cg_array,
    const RandomManager& random_manager,
    const TimerManager& timer_manager,
    const bool& inverted_pool_driver,
    const bool& allow_multiplicity,
    const bool& allow_self_connections,
    const bool& partition_connections_by_source
)
{
    const auto thread_timer_registry = timer_manager.get_thread_registry();

    const auto node_disp_timer = thread_timer_registry->get_register_timer(
        "compute_node_displacements_time"
    );
    node_disp_timer->start();

    compute_displacement_checks_across_tile_pairs(
        rpi,
        mc_array,
        inverted_pool_driver
    );

    node_disp_timer->stop();

    const auto conn_gen_timer = thread_timer_registry->get_register_timer(
        "connection_generation_time"
    );
    conn_gen_timer->start();

    compute_connections_across_tiles(
        tci,
        rpi,
        target_rank,
        cg_array,
        random_manager,
        inverted_pool_driver,
        allow_multiplicity,
        allow_self_connections,
        partition_connections_by_source
    );

    conn_gen_timer->stop();
}


template < typename CoordT >
RankConnectionInfo
compute_distributed_spatial_connections(
    const DistributedTiledNodeSequenceMap& dist_tns_source,
    const DistributedTiledNodeSequenceMap& dist_tns_target,
    const TileGrid< CoordT >& tile_grid,
    const GridNeighborhood& grid_neighborhood,
    const GridNodeCollection< CoordT >& grid_node_col,
    const TAArray< MaskCollection< CoordT > >& mc_array,
    const TAArray< ConnectionGenerator >& cg_array,
    const RandomManager& random_manager,
    const TimerManager& timer_manager,
    const bool& edge_wrap,
    const bool& only_neighborhood,
    const bool& inverted_connection_rule,
    const bool& allow_multiplicity,
    const bool& allow_self_connections,
    const bool& partition_connections_by_source
)
{
    assert( timer_manager.is_initialized() );

    if ( !mc_array.get_local_thread_item()->has_blueprint() )
        throw std::invalid_argument( "Spatial connections require mask blueprints" );

    const auto rank_timer_registry = timer_manager.get_rank_registry();

    const auto connect_timer = rank_timer_registry->get_register_timer(
        "compute_distributed_spatial_connections_time"
    );
    connect_timer->start();

    DistributedPairInfo< CoordT > dpi;
    RankConnectionInfo rci;

#pragma omp parallel default( none )\
    shared( dpi, rci, dist_tns_source, dist_tns_target,\
        tile_grid, grid_neighborhood, grid_node_col,\
        mc_array, cg_array, random_manager, timer_manager )\
    firstprivate( rank_timer_registry, edge_wrap, only_neighborhood, inverted_connection_rule,\
        allow_multiplicity, allow_self_connections, partition_connections_by_source )
#pragma omp master
#pragma omp taskgroup
    {
        const auto tile_overlap_timer = rank_timer_registry->get_register_timer(
            "compute_distributed_tile_overlap_time"
        );
        tile_overlap_timer->start();

        const auto [has_targets, has_local_targets] =
            compute_distributed_tile_overlap(
            dpi,
            dist_tns_source,
            dist_tns_target,
            tile_grid,
            grid_neighborhood,
            grid_node_col,
            mc_array,
            edge_wrap,
            only_neighborhood,
            inverted_connection_rule
        );

        tile_overlap_timer->stop();

        const auto node_overlap_timer = rank_timer_registry->get_register_timer(
            "compute_distributed_node_overlap_time"
        );
        node_overlap_timer->start();

        if ( has_targets )
        {
            if ( has_local_targets )
            {
                const auto ssi_it = dpi.source_side_info_.find( grid_neighborhood.local_rank_ );
                const auto tsi_it = dpi.target_side_info_.find( grid_neighborhood.local_rank_ );
                assert(
                    ssi_it != dpi.source_side_info_.end() &&
                    tsi_it != dpi.target_side_info_.end() &&
                    ssi_it->second.sender_info_.data_payload_.empty() &&
                    tsi_it->second.sender_info_.data_payload_.empty()
                );

                ssi_it->second.receiver_info_ =
                    std::move( tsi_it->second.sender_info_ );

                const auto emplace_it = rci.outgoing_connections_.emplace(
                    std::make_pair(
                        vp_t( grid_neighborhood.local_rank_ ),
                        TileConnectionInfo()
                    )
                ).first;

#pragma omp task default( none )\
                shared( mc_array, cg_array, random_manager, timer_manager )\
                firstprivate( ssi_it, emplace_it,\
                    inverted_connection_rule, allow_multiplicity,\
                    allow_self_connections, partition_connections_by_source )
                compute_target_connections(
                    ssi_it->second,
                    emplace_it->second,
                    ssi_it->first,
                    mc_array,
                    cg_array,
                    random_manager,
                    timer_manager,
                    inverted_connection_rule,
                    allow_multiplicity,
                    allow_self_connections,
                    partition_connections_by_source
                );
            }

            if ( grid_neighborhood.num_processes_ > 1 )
            {
                const auto communicate_timer = rank_timer_registry->get_register_timer(
                    "communicate_node_positions_time"
                );
                communicate_timer->start();

#pragma omp taskgroup
                communicate_connect_distributed_pair_data(
                    rci, dpi, mc_array, cg_array,
                    random_manager, timer_manager,
                    grid_neighborhood.local_rank_,
                    inverted_connection_rule,
                    allow_multiplicity,
                    partition_connections_by_source,
                    has_local_targets
                );

                communicate_timer->stop();
            }

            node_overlap_timer->stop();
        }
    }

    connect_timer->stop();

    return rci;
}


#ifdef HAVE_MPI
struct QueuedRequests
{
    std::unordered_set< vp_t > pending_ranks_;
    std::unordered_set< vp_t > done_ranks_;

    QueuedRequests() = default;
    QueuedRequests( const QueuedRequests& ) = delete;
    QueuedRequests( QueuedRequests&& ) = default;
};


template< typename CoordT >
void communicate_payload_async(
    QueuedRequests& request_queue,
    std::vector< MPI_Request >::iterator& request_it,
    std::unordered_map< vp_t, RankPairInfo< CoordT > >& rank_info_map,
    const vp_t& local_rank,
    const vp_t& tag
)
{
    for ( const auto& [rank, rpi] : rank_info_map )
    {
        if ( rank == local_rank ) continue;
        assert(
            request_queue.pending_ranks_.find( rank ) == request_queue.pending_ranks_.end() &&
            rpi.sender_info_.data_payload_.size() < std::numeric_limits< vp_t >::max()
        );
        const auto payload_size = static_cast< vp_t >( rpi.sender_info_.data_payload_.size() );
        assert( 3 <= payload_size );

        MPI_Isend(
            rpi.sender_info_.data_payload_.data(),
            payload_size,
            MPI_INT64_T,
            rank,
            tag,
            MPI_COMM_WORLD,
            &( *request_it++ )
        );

        request_queue.pending_ranks_.insert( rank );
    }
}


template< typename CoordT >
void probe_payload_recv(
    QueuedRequests& request_queue,
    MPI_Status& status,
    std::unordered_map< vp_t, RankPairInfo< CoordT > >& rank_info_map,
    const vp_t& tag
)
{
    if ( request_queue.pending_ranks_.empty() )
        return;

    vp_t flag;
    auto ranks_it = request_queue.pending_ranks_.cbegin();
    const auto ranks_end = request_queue.pending_ranks_.cend();
    while ( ranks_it != ranks_end )
    {
        flag = 0;
        MPI_Iprobe( *ranks_it, tag, MPI_COMM_WORLD, &flag, &status );
        if ( flag )
        {
            MPI_Get_count( &status, MPI_INT64_T, &flag );
            assert( 3 <= flag );

            const auto rank_map_it = rank_info_map.find( *ranks_it );
            assert( rank_map_it != rank_info_map.end() );
            rank_map_it->second.receiver_info_.data_payload_.resize(
                flag
            );

            MPI_Recv(
                rank_map_it->second.receiver_info_.data_payload_.data(),
                flag,
                MPI_INT64_T,
                *ranks_it,
                tag,
                MPI_COMM_WORLD,
                MPI_STATUS_IGNORE
            );

            assert( request_queue.done_ranks_.find( *ranks_it ) == request_queue.done_ranks_.end() );
            request_queue.done_ranks_.insert( *ranks_it );
            ranks_it = request_queue.pending_ranks_.erase( ranks_it );
        }
        else
            ++ranks_it;
    }
}


template < typename CoordT >
void dispatch_connection_procedures(
    QueuedRequests& request_queue,
    std::unordered_map< vp_t, RankPairInfo< CoordT > >& rank_info_map,
    std::unordered_map< vp_t, TileConnectionInfo >& connection_info_map,
    const TAArray< MaskCollection< CoordT > >& mc_array,
    const TAArray< ConnectionGenerator >& cg_array,
    const RandomManager& random_manager,
    const TimerManager& timer_manager,
    const bool& inverted_pool_driver,
    const bool& allow_multiplicity,
    const bool& partition_connections_by_source
)
{
    if ( request_queue.done_ranks_.empty() )
        return;

    auto ranks_it = request_queue.done_ranks_.cbegin();
    const auto ranks_end = request_queue.done_ranks_.cend();
    while ( ranks_it != ranks_end )
    {
        const auto rank_info_it = rank_info_map.find( *ranks_it );
        assert( rank_info_it != rank_info_map.end() );
        ranks_it = request_queue.done_ranks_.erase( ranks_it );

        const auto emplace_it = connection_info_map.emplace(
            std::make_pair(
                vp_t( rank_info_it->first ),
                TileConnectionInfo()
            )
        ).first;

#pragma omp task default( none )\
    shared( mc_array, cg_array, random_manager, timer_manager )\
    firstprivate( rank_info_it, emplace_it,\
    inverted_pool_driver, allow_multiplicity,\
    partition_connections_by_source )
        {
            reconstruct_received_info( rank_info_it->second.receiver_info_ );
            compute_target_connections(
                rank_info_it->second,
                emplace_it->second,
                rank_info_it->first,
                mc_array,
                cg_array,
                random_manager,
                timer_manager,
                inverted_pool_driver,
                allow_multiplicity,
                true, // Using distributed indexes no need to check for self connections
                partition_connections_by_source
            );
        }
    }
}
#endif


template < typename CoordT >
void communicate_connect_distributed_pair_data(
    RankConnectionInfo& rank_connection_map,
    DistributedPairInfo< CoordT >& distributed_pair_data,
    const TAArray< MaskCollection< CoordT > >& mc_array,
    const TAArray< ConnectionGenerator >& cg_array,
    const RandomManager& random_manager,
    const TimerManager& timer_manager,
    const vp_t& local_rank,
    const bool& inverted_connection_rule,
    const bool& allow_multiplicity,
    const bool& partition_connections_by_source,
    const bool& has_local_targets
)
{
    // This function is meant to be called
    // inside parallel section
    // inside master thread context
    // within a taskgroup block
    const auto source_side_size = distributed_pair_data.source_side_info_.size();
    const auto target_side_size = distributed_pair_data.target_side_info_.size();
    if (
        // First condition implies only local targets available -> return
        ( has_local_targets && ( source_side_size == 1 && target_side_size == 1 ) ) ||
        ( source_side_size == 0 && target_side_size == 0 )
        ) return;

#ifdef HAVE_MPI
    std::vector< MPI_Request > send_requests(
        source_side_size + target_side_size - has_local_targets * 2
    );
    auto req_it = send_requests.begin();

    QueuedRequests source_side_queue;
    QueuedRequests target_side_queue;

    const auto rank_timer_registry = timer_manager.get_rank_registry();

    const auto send_async_timer = rank_timer_registry->get_register_timer(
        "send_async_positions_time"
    );
    send_async_timer->start();

    if ( !distributed_pair_data.source_side_info_.empty() )
        communicate_payload_async(
            source_side_queue,
            req_it,
            distributed_pair_data.source_side_info_,
            local_rank,
            0 // tag 0 for sending source side positions
        );

    if ( !distributed_pair_data.target_side_info_.empty() )
        communicate_payload_async(
            target_side_queue,
            req_it,
            distributed_pair_data.target_side_info_,
            local_rank,
            1 // tag 1 for sending target side positions
        );

    assert( req_it == send_requests.end() );

    send_async_timer->stop();

    const auto trigger_send_timer = rank_timer_registry->get_register_timer(
        "trigger_send_timer"
    );
    trigger_send_timer->start();

    vp_t flag;
    MPI_Testall(
        send_requests.size(),
        send_requests.data(),
        &flag,
        MPI_STATUSES_IGNORE
    );

    trigger_send_timer->stop();

    const auto last_compute_dispatch_timer = rank_timer_registry->get_register_timer(
        "last_compute_dispatch_timer"
    );
    const auto first_compute_dispatch_timer = rank_timer_registry->get_register_timer(
        "first_compute_dispatch_timer"
    );

    last_compute_dispatch_timer->start();
    first_compute_dispatch_timer->start();

    MPI_Status status;
    bool first_compute_dispatch = true;
    while (
        !( source_side_queue.done_ranks_.empty() && source_side_queue.pending_ranks_.empty() ) ||
        !( target_side_queue.done_ranks_.empty() && target_side_queue.pending_ranks_.empty() )
        )
    {
        probe_payload_recv(
            source_side_queue,
            status,
            distributed_pair_data.source_side_info_,
            1 // mirrored tag for receiving
        );

        if ( !source_side_queue.done_ranks_.empty() )
        {
            dispatch_connection_procedures(
                source_side_queue,
                distributed_pair_data.source_side_info_,
                inverted_connection_rule
                ? rank_connection_map.incoming_connections_
                : rank_connection_map.outgoing_connections_,
                mc_array,
                cg_array,
                random_manager,
                timer_manager,
                inverted_connection_rule,
                allow_multiplicity,
                partition_connections_by_source
            );

            if ( first_compute_dispatch )
            {
                first_compute_dispatch_timer->stop();
                first_compute_dispatch = false;
            }
        }

        probe_payload_recv(
            target_side_queue,
            status,
            distributed_pair_data.target_side_info_,
            0 // mirrored tag for receiving
        );

        if ( !target_side_queue.done_ranks_.empty() )
        {
            dispatch_connection_procedures(
                target_side_queue,
                distributed_pair_data.target_side_info_,
                inverted_connection_rule
                ? rank_connection_map.outgoing_connections_
                : rank_connection_map.incoming_connections_,
                mc_array,
                cg_array,
                random_manager,
                timer_manager,
                !inverted_connection_rule,
                allow_multiplicity,
                partition_connections_by_source
            );

            if ( first_compute_dispatch )
            {
                first_compute_dispatch_timer->stop();
                first_compute_dispatch = false;
            }
        }
    }

    last_compute_dispatch_timer->stop();

    const auto wait_send_timer = rank_timer_registry->get_register_timer(
        "wait_send_timer"
    );
    wait_send_timer->start();

    MPI_Waitall(
        send_requests.size(),
        send_requests.data(),
        MPI_STATUSES_IGNORE
    );

    // At this point it is safe to clear sender payloads
    for ( const auto& rpi_map_ptr :
        {
            &distributed_pair_data.source_side_info_,
            &distributed_pair_data.target_side_info_
        } )
        for ( auto& rpi_pair : *rpi_map_ptr )
            rpi_pair.second.sender_info_.data_payload_.clear();

    wait_send_timer->stop();
#endif
}
}


#endif
