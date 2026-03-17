/*
 *  connection_generation.cpp
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

#include "connection_generation.h"


namespace sapi
{
void consolidate_connection_map(
    TileConnectionInfo& tci,
    const bool& partition_connections_by_source
)
{
    if ( tci.total_generated_connections_ < 1 )
        return;

    if ( partition_connections_by_source )
    {
        std::unordered_map< conn_index_t, std::size_t >
            source_split_indexes;

        std::unordered_map< std::size_t,
            std::unordered_map< conn_index_t,
            std::tuple< conn_index_t, conn_param_t, conn_param_t > > >
            source_split_connection_maps;

        source_split_connection_maps.emplace(
            std::make_pair(
                0,
                std::unordered_map< conn_index_t,
                std::tuple< conn_index_t, conn_param_t, conn_param_t > >()
            )
        );

        bool at_least_one_emplaced = false;
        auto ci_fl_move_it = std::make_move_iterator( tci.procedural_connection_list_.begin() );
        while ( !tci.procedural_connection_list_.empty() )
        {
            auto ci_fl = *ci_fl_move_it++;
            tci.procedural_connection_list_.pop_front();

            auto ci_move_it = std::make_move_iterator( ci_fl.begin() );
            while ( !ci_fl.empty() )
            {
                const auto [driver_index, pool_index, weight, delay, multiplicity] = *ci_move_it++;
                ci_fl.pop_front();

                std::size_t next_map_index = 0;
                auto conn_info = std::make_tuple( pool_index, weight, delay );
                auto driver_search = source_split_indexes.find( driver_index );
                if ( driver_search == source_split_indexes.end() )
                {
                    driver_search = source_split_indexes.emplace(
                        std::make_pair(
                            conn_index_t( driver_index ),
                            0
                        )
                    ).first;
                }
                else
                    next_map_index = driver_search->second;

                for ( mult_t mult = 0; mult < multiplicity; ++mult )
                {
                    const auto one_after_last_index = source_split_connection_maps.size();
                    if ( next_map_index < one_after_last_index )
                    {
                        const auto emplace_it = source_split_connection_maps[ next_map_index ].emplace(
                            std::make_pair(
                                conn_index_t( driver_index ),
                                std::move( conn_info )
                            )
                        );
                        assert( emplace_it.second );
                    }
                    else
                    {
                        std::unordered_map< conn_index_t,
                            std::tuple< conn_index_t, conn_param_t, conn_param_t > > next_map;
                        next_map.emplace(
                            std::make_pair(
                                conn_index_t( driver_index ),
                                std::move( conn_info )
                            )
                        );

                        const auto emplace_it = source_split_connection_maps.emplace(
                            std::make_pair(
                                std::size_t( one_after_last_index ),
                                std::move( next_map )
                            )
                        );
                        assert( emplace_it.second );
                    }
                    ++next_map_index;
                }

                driver_search->second = next_map_index;
                at_least_one_emplaced = true;
            }
        }

        assert( at_least_one_emplaced );

        count_t emplaced_connections = 0;
        tci.partitioned_connection_vectors_.reserve( source_split_connection_maps.size() );
        for ( const auto& split_conn_map : source_split_connection_maps )
        {
            assert( !split_conn_map.second.empty() );
            ConnectionVectors cvec;
            // Conversion guaranteed by total_generated_connections being positive
            cvec.prepare_vectors( static_cast< count_t >( split_conn_map.second.size() ) );
            emplaced_connections += cvec.sizes_;
            for ( const auto& [driver_index, conn_tuple] : split_conn_map.second )
            {
                const auto& [pool_index, weight, delay] = conn_tuple;

                cvec.connection_sources_.emplace_back( driver_index );
                cvec.connection_targets_.emplace_back( pool_index );
                cvec.connection_weights_.emplace_back( weight );
                cvec.connection_delays_.emplace_back( delay );
            }
            tci.partitioned_connection_vectors_.emplace_back( std::move( cvec ) );
        }

        assert( emplaced_connections == tci.total_generated_connections_ );
    }
    else
    {
        tci.partitioned_connection_vectors_.resize( 1 );
        const auto conn_vec = tci.partitioned_connection_vectors_.begin();
        conn_vec->prepare_vectors( tci.total_generated_connections_ );
        auto ci_fl_move_it = std::make_move_iterator( tci.procedural_connection_list_.begin() );
        while ( !tci.procedural_connection_list_.empty() )
        {
            auto ci_fl = *ci_fl_move_it++;
            tci.procedural_connection_list_.pop_front();

            auto ci_move_it = std::make_move_iterator( ci_fl.begin() );
            while ( !ci_fl.empty() )
            {
                const auto [driver_index, pool_index, weight, delay, multiplicity] = *ci_move_it++;
                ci_fl.pop_front();

                for ( mult_t mult = 0; mult < multiplicity; ++mult )
                {
                    conn_vec->connection_sources_.emplace_back( driver_index );
                    conn_vec->connection_targets_.emplace_back( pool_index );
                    conn_vec->connection_weights_.emplace_back( weight );
                    conn_vec->connection_delays_.emplace_back( delay );
                }
            }
        }

        assert( conn_vec->connection_sources_.size() == static_cast< std::size_t >( tci.total_generated_connections_ ) );
    }
}
}
