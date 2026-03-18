/*
 *  connection_containers.h
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

#ifndef CONNECTION_CONTAINERS_H
#define CONNECTION_CONTAINERS_H

#include <map>
#include <tuple>
#include <vector>
#include <limits>
#include <forward_list>
#include <unordered_map>

#include "sapi_config.h"


namespace sapi
{
typedef std::tuple<
    conn_index_t, // source
    conn_index_t, // target
    conn_param_t, // weight
    conn_param_t, // delay
    mult_t        // multiplicity
> ConnectionInfo;


struct ConnectionVectors
{
    count_t sizes_ = 0;
    conn_index_t first_index_ = std::numeric_limits< conn_index_t >::max();

    std::vector< conn_index_t > connection_sources_;
    std::vector< conn_index_t > connection_targets_;
    std::vector< conn_param_t > connection_weights_;
    std::vector< conn_param_t > connection_delays_;

    ConnectionVectors() = default;
    ConnectionVectors( const ConnectionVectors& ) = delete;
    ConnectionVectors( ConnectionVectors&& ) = default;
    ~ConnectionVectors() = default;

    void prepare_vectors( const count_t& size );
};


// Maps are aggregated by driver tile/node
struct TileConnectionInfo
{
    count_t total_generated_connections_ = 0;

    std::forward_list< std::forward_list< ConnectionInfo > > procedural_connection_list_;

    std::vector< ConnectionVectors > partitioned_connection_vectors_;

    TileConnectionInfo() = default;
    TileConnectionInfo( const TileConnectionInfo& ) = delete;
    TileConnectionInfo( TileConnectionInfo&& ) = default;
    ~TileConnectionInfo() = default;

    bool operator==( const TileConnectionInfo& tci ) const
    {
        return build_connection_map() == tci.build_connection_map();
    }

    std::map< conn_index_t,
        std::map< conn_index_t,
        std::tuple< conn_param_t, conn_param_t, mult_t > > >
        build_connection_map() const;
};


struct RankConnectionInfo
{
    std::unordered_map< vp_t, TileConnectionInfo > outgoing_connections_;
    std::unordered_map< vp_t, TileConnectionInfo > incoming_connections_;

    RankConnectionInfo() = default;
    RankConnectionInfo( const RankConnectionInfo& ) = delete;
    RankConnectionInfo( RankConnectionInfo&& rci ) = default;
    ~RankConnectionInfo() = default;

    bool operator==( const RankConnectionInfo& rci ) const
    {
        return outgoing_connections_ == rci.outgoing_connections_
            && incoming_connections_ == rci.incoming_connections_;
    }
};
}


#endif
