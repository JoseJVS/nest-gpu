/*
 *  node_slicing.cpp
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

#include "node_slicing.h"


namespace sapi
{
TiledNodeSequenceList
get_optional_tiled_node_sequences(
    const std::optional< DistTns_IT >& opt_dist_tns_it,
    const GridNeighborhood& grid_neighborhood
)
{
    TiledNodeSequenceList optional_tns;

    if ( opt_dist_tns_it.has_value() )
    {
        auto tns_it = opt_dist_tns_it.value()->second.cbegin();
        const auto tns_end = opt_dist_tns_it.value()->second.cend();
        for ( ; tns_it != tns_end; ++tns_it )
            optional_tns.emplace_front(
                tns_it->first,
                std::make_optional( tns_it )
            );
    }
    else
    {
        for ( const auto& position : *grid_neighborhood.locally_owned_tiles_ )
            optional_tns.emplace_front(
                position,
                std::optional< Tns_IT >()
            );
    }

    return optional_tns;
}
}
