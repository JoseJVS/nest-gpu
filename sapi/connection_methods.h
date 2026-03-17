/*
 *  connection_methods.h
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

#ifndef CONNECTION_METHODS_H
#define CONNECTION_METHODS_H

#include <random>

#include "enum_store.h"
#include "mask_containers.h"
#include "numeric_functors.h"
#include "type_erasure_helpers.h"
#include "connection_containers.h"


namespace sapi
{
template < typename CoordT >
inline bool remove_self_target(
    std::map< nodeidx_t, Displacement< CoordT > >& pool_map,
    const nodeidx_t& driver_index,
    const bool& allow_self_connections
)
{
    if ( allow_self_connections )
        return false;

    if ( const auto self_search =
        pool_map.find( driver_index );
        self_search != pool_map.end() )
    {
        pool_map.erase( self_search );
        if ( pool_map.empty() )
            return true;
    }

    return false;
}


template < typename DistributionT >
inline mult_t draw_connection_multiplicity(
    AnyRNG& rng,
    DistributionT& dist,
    const space_t& probability_value,
    const bool& allow_multiplicity
)
{
    if constexpr ( std::is_same_v< DistributionT, std::uniform_real_distribution< space_t > > )
    {
        return std::isless( dist( rng ), probability_value );
    }
    else
    {
        dist.param(
            static_cast< typename DistributionT::param_type >( probability_value )
        );
        return allow_multiplicity
            ? static_cast< mult_t >( dist( rng ) )
            : static_cast< mult_t >( std::isless( 0, dist( rng ) ) );
    }
}


template < typename CoordT, typename DistributionT >
void compute_probabilistic_connections(
    TileConnectionInfo& tci,
    std::forward_list< ConnectionInfo >& conn_list,
    ConsolidatedNodeDisplacementMap< CoordT >& displacement_map,
    AnyRNG& rng,
    DistributionT& dist,
    const NFCollection& cfc,
    const mult_t& max_connections,
    const bool& allow_multiplicity,
    const bool& allow_self_connections
)
{
    assert(
        conn_list.empty() &&
        !displacement_map.empty() &&
        cfc.weight_functor_.is_initialized() &&
        cfc.delay_functor_.is_initialized() &&
        cfc.probability_functor_.is_initialized()
    );

    count_t total_connections = 0;
    const bool track_connections = 0 < max_connections;
    for ( auto& [driver_index, pool_map] : displacement_map )
    {
        assert( !pool_map.empty() );

        if ( remove_self_target(
            pool_map, driver_index, allow_self_connections
        ) ) continue;

        mult_t driver_connections = 0;
        for ( auto& [pool_index, displacement] : pool_map )
        {
            if ( track_connections && max_connections <= driver_connections )
                break;

            auto multiplicity = draw_connection_multiplicity(
                rng,
                dist,
                cfc.probability_functor_( displacement ),
                allow_multiplicity
            );

            if ( multiplicity < 1 ) continue;

            if ( track_connections )
            {
                if ( ( driver_connections + multiplicity ) <= max_connections )
                {
                    driver_connections += multiplicity;
                }
                else
                {
                    multiplicity = max_connections - driver_connections;
                    driver_connections = max_connections;
                }
            }

            total_connections += multiplicity;

            conn_list.emplace_front(
                ConnectionInfo(
                    conn_index_t( driver_index ),
                    conn_index_t( pool_index ),
                    conn_param_t( cfc.weight_functor_( displacement ) ),
                    conn_param_t( cfc.delay_functor_( displacement ) ),
                    mult_t( multiplicity )
                )
            );
        }

        pool_map.clear();
    }

    assert( 0 <= total_connections );

    displacement_map.clear();

    if ( 0 < total_connections )
#pragma omp atomic
        tci.total_generated_connections_ += total_connections;
}


std::vector< mult_t >
generate_connection_counts(
    AnyRNG& rng,
    const std::size_t& available_targets,
    const mult_t& expected_connection_count,
    const bool& allow_multiplicity
);


template < typename CoordT >
void compute_fixed_number_connections(
    TileConnectionInfo& tci,
    std::forward_list< ConnectionInfo >& conn_list,
    ConsolidatedNodeDisplacementMap< CoordT >& displacement_map,
    AnyRNG& rng,
    const NFCollection& cfc,
    const mult_t& expected_connection_counts,
    const bool& allow_multiplicity,
    const bool& allow_self_connections
)
{
    assert(
        conn_list.empty() &&
        !displacement_map.empty() &&
        cfc.weight_functor_.is_initialized() &&
        cfc.delay_functor_.is_initialized() &&
        !cfc.probability_functor_.is_initialized()
    );

    if ( expected_connection_counts == 0 )
    {
        displacement_map.clear();
        return;
    }

    count_t total_connections = 0;
    for ( auto& [driver_index, pool_map] : displacement_map )
    {
        assert( !pool_map.empty() );

        if ( remove_self_target(
            pool_map, driver_index, allow_self_connections
        ) ) continue;

        const auto connection_counts = generate_connection_counts(
            rng,
            pool_map.size(),
            expected_connection_counts,
            allow_multiplicity
        );

        auto connection_count_it = connection_counts.cbegin();
        for ( auto& [pool_index, displacement] : pool_map )
        {
            const auto multiplicity = *connection_count_it++;
            if ( multiplicity < 1 ) continue;

            total_connections += multiplicity;

            conn_list.emplace_front(
                ConnectionInfo(
                    conn_index_t( driver_index ),
                    conn_index_t( pool_index ),
                    conn_param_t( cfc.weight_functor_( displacement ) ),
                    conn_param_t( cfc.delay_functor_( displacement ) ),
                    mult_t( multiplicity )
                )
            );
        }

        pool_map.clear();
    }

    assert( 0 <= total_connections );

    displacement_map.clear();

    if ( 0 < total_connections )
#pragma omp atomic
        tci.total_generated_connections_ += total_connections;
}


template < typename CoordT >
inline void compute_connections(
    TileConnectionInfo& tci,
    std::forward_list< ConnectionInfo >& conn_list,
    ConsolidatedNodeDisplacementMap< CoordT >& displacement_map,
    AnyRNG& rng,
    const NFCollection& cfc,
    const mult_t& connection_counts,
    const bool& allow_multiplicity,
    const bool& allow_self_connections,
    const CONNECTION_METHOD& cm
)
{
    switch ( cm )
    {
    case CONNECTION_METHOD::FIXED_NUMBER:
        compute_fixed_number_connections(
            tci,
            conn_list,
            displacement_map,
            rng,
            cfc,
            connection_counts,
            allow_multiplicity,
            allow_self_connections
        );
        break;

    case CONNECTION_METHOD::PAIRWISE_BERNOULLI:
    {
        std::uniform_real_distribution< space_t > dist( 0., 1. );
        compute_probabilistic_connections(
            tci,
            conn_list,
            displacement_map,
            rng,
            dist,
            cfc,
            connection_counts,
            allow_multiplicity,
            allow_self_connections
        );
        break;
    }

    case CONNECTION_METHOD::PAIRWISE_POISSON:
    {
        std::poisson_distribution< mult_t > dist;
        compute_probabilistic_connections(
            tci,
            conn_list,
            displacement_map,
            rng,
            dist,
            cfc,
            connection_counts,
            allow_multiplicity,
            allow_self_connections
        );
        break;
    }

    default:
        std::invalid_argument( "Invalid connection method" );
    }
}
}


#endif
