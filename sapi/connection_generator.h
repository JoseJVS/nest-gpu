/*
 *  connection_generator.h
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

#ifndef CONNECTION_GENERATOR_H
#define CONNECTION_GENERATOR_H

#include "connection_methods.h"


namespace sapi
{
struct ConnectionGenerator final : public Cloneable< ConnectionGenerator >
{
    CONNECTION_METHOD cm_ = CONNECTION_METHOD::NULL_CM;
    mult_t connection_counts_ = 0;
    NFCollection cfc_;

    ConnectionGenerator() = default;
    ConnectionGenerator( const ConnectionGenerator& ) = default;
    ConnectionGenerator( ConnectionGenerator&& ) = default;

    ConnectionGenerator( const CONNECTION_METHOD& cm )
        : cm_( cm )
    {
    }

    ConnectionGenerator& operator=( ConnectionGenerator&& );

    std::unique_ptr< ConnectionGenerator >
        clone() const override;

    template < typename CoordT >
    void compute_connections(
        TileConnectionInfo& tci,
        std::forward_list< ConnectionInfo >& conn_list,
        ConsolidatedNodeDisplacementMap< CoordT >& displacement_map,
        AnyRNG& rng,
        const bool& allow_multiplicity,
        const bool& allow_self_connections
    ) const;
};


inline ConnectionGenerator&
ConnectionGenerator::operator=( ConnectionGenerator&& cg )
{
    cm_ = cg.cm_;
    connection_counts_ = cg.connection_counts_;
    cfc_ = std::move( cg.cfc_ );

    cg.cm_ = CONNECTION_METHOD::NULL_CM;

    return *this;
}


inline std::unique_ptr< ConnectionGenerator >
ConnectionGenerator::clone() const
{
    return std::make_unique< ConnectionGenerator >( *this );
}


template < typename CoordT >
inline void ConnectionGenerator::compute_connections(
    TileConnectionInfo& tci,
    std::forward_list< ConnectionInfo >& conn_list,
    ConsolidatedNodeDisplacementMap< CoordT >& displacement_map,
    AnyRNG& rng,
    const bool& allow_multiplicity,
    const bool& allow_self_connections
) const
{
    sapi::compute_connections(
        tci,
        conn_list,
        displacement_map,
        rng,
        cfc_,
        connection_counts_,
        allow_multiplicity,
        allow_self_connections,
        cm_
    );
}
}


#endif
