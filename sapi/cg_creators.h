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
