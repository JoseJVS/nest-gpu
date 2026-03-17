/*
 *  connection_methods.cpp
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

#include "connection_methods.h"


namespace sapi
{
std::vector< mult_t >
    generate_connection_counts(
        AnyRNG& rng,
        const std::size_t& available_targets,
        const mult_t& expected_connection_count,
        const bool& allow_multiplicity
    )
{
    assert( 0 < available_targets && 0 < expected_connection_count );

    std::vector< mult_t > connection_counts;
    auto connections_todo = expected_connection_count;

    if ( static_cast< std::size_t >( expected_connection_count ) < available_targets )
    {
        connection_counts.resize( available_targets, 0 );
    }
    else
    {
        if ( available_targets == 1 )
        {
            connection_counts.resize( 1,
                allow_multiplicity ? expected_connection_count : 1 );
            connections_todo = 0;
        }
        else
        {
            connection_counts.resize( available_targets, !allow_multiplicity );
            connections_todo = static_cast< mult_t >( allow_multiplicity ) * expected_connection_count;
        }
    }

    if ( 0 < connections_todo && allow_multiplicity )
    {
        std::uniform_int_distribution< std::size_t > dist( 0, available_targets - 1 );
        while ( 0 < connections_todo )
        {
            ++connection_counts[ dist( rng ) ];
            --connections_todo;
        }
    }
    else if ( 0 < connections_todo )
    {
        std::vector< nodeidx_t > indexes( available_targets );
        std::iota( indexes.begin(), indexes.end(), 0 );
        std::shuffle( indexes.begin(), indexes.end(), rng );

        while ( 0 < connections_todo )
        {
            ++connection_counts[ indexes[ connections_todo ] ];
            --connections_todo;
        }
    }

    assert( connections_todo < expected_connection_count );

    return connection_counts;
}
}
