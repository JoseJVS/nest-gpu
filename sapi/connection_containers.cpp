/*
 *  connection_containers.cpp
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

#include <cassert>

#include "connection_containers.h"


namespace sapi
{
void ConnectionVectors::prepare_vectors( const count_t& size )
{
    assert(
        0 <= size &&
        connection_sources_.empty() &&
        connection_targets_.empty() &&
        connection_weights_.empty() &&
        connection_delays_.empty()
    );

    if ( size == 0 )
        return;

    connection_sources_.reserve( size );
    connection_targets_.reserve( size );
    connection_weights_.reserve( size );
    connection_delays_.reserve( size );

    sizes_ = size;
}


std::map< conn_index_t,
    std::map< conn_index_t,
    std::tuple< conn_param_t, conn_param_t, mult_t > > >
    TileConnectionInfo::build_connection_map() const
{
    std::map< conn_index_t,
        std::map< conn_index_t,
        std::tuple< conn_param_t, conn_param_t, mult_t > > > conn_map;

    for ( const auto& conn_vec : partitioned_connection_vectors_ )
    {
        for ( count_t conn_idx = 0; conn_idx < conn_vec.sizes_; ++conn_idx )
        {
            const auto source = conn_vec.connection_sources_[ conn_idx ];
            const auto target = conn_vec.connection_targets_[ conn_idx ];
            const auto weight = conn_vec.connection_weights_[ conn_idx ];
            const auto delay = conn_vec.connection_delays_[ conn_idx ];

            auto source_search = conn_map.find( source );
            if ( source_search == conn_map.end() )
                source_search = conn_map.emplace(
                    std::make_pair(
                        source,
                        std::map< conn_index_t,
                        std::tuple< conn_param_t, conn_param_t, mult_t > >()
                    )
                ).first;

            auto target_search = source_search->second.find( target );
            if ( target_search == source_search->second.end() )
                target_search = source_search->second.emplace(
                    std::make_pair(
                        target,
                        std::make_tuple(
                            weight,
                            delay,
                            1
                        )
                    )
                ).first;
            else
                std::get< 2 >( target_search->second ) += 1;
        }
    }

    return conn_map;
}
}
