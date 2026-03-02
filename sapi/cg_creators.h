/*
 *  cg_creators.h
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

#ifndef CG_CREATORS_H
#define CG_CREATORS_H

#include "connection_generators.h"


namespace sapi
{
// Forward definitions to link with coordinates.h
struct Coord2D;
struct Coord3D;


template < typename CoordT >
struct PairWiseBernoulliCGCreator : public StateLessCreator< ConnectionGenerator< CoordT > >
{
    std::unique_ptr< ConnectionGenerator< CoordT > > create() const override
    {
        return std::make_unique< PairWiseBernoulliPCG< CoordT > >();
    }
};


template < typename CoordT >
struct PairWisePoissonCGCreator : public StateLessCreator< ConnectionGenerator< CoordT > >
{
    std::unique_ptr< ConnectionGenerator< CoordT > > create() const override
    {
        return std::make_unique< PairWisePoissonPCG< CoordT > >();
    }
};


template < typename CoordT >
struct FixedNumberCGCreator : public StateLessCreator< ConnectionGenerator< CoordT > >
{
    std::unique_ptr< ConnectionGenerator< CoordT > > create() const override
    {
        return std::make_unique< FixedNumberCG< CoordT > >();
    }
};


inline void initialize_cg_registry( CreatorRegistry< ConnectionGenerator< Coord2D > >& cgr )
{
    cgr.register_creator< PairWiseBernoulliCGCreator< Coord2D > >
        ( "PairWiseBernoulli" );
    cgr.register_creator< PairWisePoissonCGCreator< Coord2D > >
        ( "PairWisePoisson" );
    cgr.register_creator< FixedNumberCGCreator< Coord2D > >
        ( "FixedNumber" );
}


inline void initialize_cg_registry( CreatorRegistry< ConnectionGenerator< Coord3D > >& cgr )
{
    cgr.register_creator< PairWiseBernoulliCGCreator< Coord3D > >
        ( "PairWiseBernoulli" );
    cgr.register_creator< PairWisePoissonCGCreator< Coord3D > >
        ( "PairWisePoisson" );
    cgr.register_creator< FixedNumberCGCreator< Coord3D > >
        ( "FixedNumber" );
}
}


#endif
