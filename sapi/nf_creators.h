/*
 *  nf_creators.h
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

#ifndef CF_CREATORS_H
#define CF_CREATORS_H

#include "numeric_functors.h"
#include "creator_registry.h"


namespace sapi
{
// Forward definitions to link with coordinates.h
struct Coord2D;
struct Coord3D;


template < typename CoordT >
struct DistanceDFCreator : public StateLessCreator< DisplacementFunctor< CoordT > >
{
    std::unique_ptr< DisplacementFunctor< CoordT > > create(
        const std::vector< space_t >& params
    ) const override
    {
        if ( !params.empty() )
            throw std::invalid_argument( "Incorrect DistanceDF params" );
        return std::make_unique< DistanceDF< CoordT > >();
    }
};


template < typename CoordT >
struct ConstantDFCreator : public StateLessCreator< DisplacementFunctor< CoordT > >
{
    std::unique_ptr< DisplacementFunctor< CoordT > > create(
        const std::vector< space_t >& params
    ) const override
    {
        if ( 1 < params.size() )
            throw std::invalid_argument( "Incorrect ConstantDF params" );
        if ( params.empty() )
            return std::make_unique< ConstantDF< CoordT > >();
        else
            return std::make_unique< ConstantDF< CoordT > >( params[ 0 ] );
    }
};


struct IdentityUFCreator : public StateLessCreator< UnaryFunctor >
{
    std::unique_ptr< UnaryFunctor > create(
        const std::vector< space_t >& params
    ) const override
    {
        if ( !params.empty() )
            throw std::invalid_argument( "Incorrect IdentityUF params" );
        return std::make_unique< IdentityUF >();
    }
};


struct MinUFCreator : public StateLessCreator< UnaryFunctor >
{
    std::unique_ptr< UnaryFunctor > create(
        const std::vector< space_t >& params
    ) const override
    {
        if ( 1 < params.size() )
            throw std::invalid_argument( "Incorrect MinUF params" );
        if ( params.empty() )
            return std::make_unique< MinUF >();
        else
            return std::make_unique< MinUF >( params[ 0 ] );
    }
};


struct MaxUFCreator : public StateLessCreator< UnaryFunctor >
{
    std::unique_ptr< UnaryFunctor > create(
        const std::vector< space_t >& params
    ) const override
    {
        if ( 1 < params.size() )
            throw std::invalid_argument( "Incorrect MaxUF params" );
        if ( params.empty() )
            return std::make_unique< MaxUF >();
        else
            return std::make_unique< MaxUF >( params[ 0 ] );
    }
};


struct InverseUFCreator : public StateLessCreator< UnaryFunctor >
{
    std::unique_ptr< UnaryFunctor > create(
        const std::vector< space_t >& params
    ) const override
    {
        if ( !params.empty() )
            throw std::invalid_argument( "Incorrect InverseUF params" );
        return std::make_unique< InverseUF >();
    }
};


struct ProportionalUFCreator : public StateLessCreator< UnaryFunctor >
{
    std::unique_ptr< UnaryFunctor > create(
        const std::vector< space_t >& params
    ) const override
    {
        if ( 1 < params.size() )
            throw std::invalid_argument( "Incorrect ProportionalUF params" );
        if ( params.empty() )
            return std::make_unique< ProportionalUF >();
        else
            return std::make_unique< ProportionalUF >( params[ 0 ] );
    }
};


struct UpperBoundUFCreator : public StateLessCreator< UnaryFunctor >
{
    std::unique_ptr< UnaryFunctor > create(
        const std::vector< space_t >& params
    ) const override
    {
        if ( !( params.empty() || 2 == params.size() ) )
            throw std::invalid_argument( "Incorrect UpperBoundUF params" );
        if ( params.empty() )
            return std::make_unique< UpperBoundUF >();
        else
            return std::make_unique< UpperBoundUF >( params[ 0 ], params[ 1 ] );
    }
};


struct LowerBoundUFCreator : public StateLessCreator< UnaryFunctor >
{
    std::unique_ptr< UnaryFunctor > create(
        const std::vector< space_t >& params
    ) const override
    {
        if ( !( params.empty() || 2 == params.size() ) )
            throw std::invalid_argument( "Incorrect LowerBoundUF params" );
        if ( params.empty() )
            return std::make_unique< LowerBoundUF >();
        else
            return std::make_unique< LowerBoundUF >( params[ 0 ], params[ 1 ] );
    }
};


struct OffsetUFCreator : public StateLessCreator< UnaryFunctor >
{
    std::unique_ptr< UnaryFunctor > create(
        const std::vector< space_t >& params
    ) const override
    {
        if ( 1 < params.size() )
            throw std::invalid_argument( "Incorrect OffsetUF params" );
        if ( params.empty() )
            return std::make_unique< OffsetUF >();
        else
            return std::make_unique< OffsetUF >( params[ 0 ] );
    }
};


struct ExponentialUFCreator : public StateLessCreator< UnaryFunctor >
{
    std::unique_ptr< UnaryFunctor > create(
        const std::vector< space_t >& params
    ) const override
    {
        if ( 1 < params.size() )
            throw std::invalid_argument( "Incorrect ExponentialUF params" );
        if ( params.empty() )
            return std::make_unique< ExponentialUF >();
        else
        {
            if ( almost_zero( params[ 0 ] ) )
                throw std::invalid_argument( "Incorrect ExponentialUF params" );
            return std::make_unique< ExponentialUF >( params[ 0 ] );
        }
    }
};


struct GaussianUFCreator : public StateLessCreator< UnaryFunctor >
{
    std::unique_ptr< UnaryFunctor > create(
        const std::vector< space_t >& params
    ) const override
    {
        if ( !( params.empty() || 2 == params.size() ) )
            throw std::invalid_argument( "Incorrect GaussianUF params" );
        if ( params.empty() )
            return std::make_unique< GaussianUF >();
        else
        {
            const auto std2 = squared( params[ 1 ] );
            if ( almost_zero( std2 ) )
                throw std::invalid_argument( "Incorrect GaussianUF params" );
            return std::make_unique< GaussianUF >( params[ 0 ], std2 );
        }
    }
};


inline void initialize_uf_registry( CreatorRegistry< UnaryFunctor >& ufr )
{
    ufr.register_creator< IdentityUFCreator >( "Identity" );
    ufr.register_creator< MinUFCreator >( "Min" );
    ufr.register_creator< MaxUFCreator >( "Max" );
    ufr.register_creator< InverseUFCreator >( "Inverse" );
    ufr.register_creator< ProportionalUFCreator >( "Proportional" );
    ufr.register_creator< UpperBoundUFCreator >( "UpperBound" );
    ufr.register_creator< LowerBoundUFCreator >( "LowerBound" );
    ufr.register_creator< OffsetUFCreator >( "Offset" );
    ufr.register_creator< ExponentialUFCreator >( "Exponential" );
    ufr.register_creator< GaussianUFCreator >( "Gaussian" );
}


inline void initialize_df_registry( CreatorRegistry< DisplacementFunctor< Coord2D > >& dfr )
{
    dfr.register_creator< DistanceDFCreator< Coord2D > >( "Distance" );
    dfr.register_creator< ConstantDFCreator< Coord2D > >( "Constant" );
}


inline void initialize_df_registry( CreatorRegistry< DisplacementFunctor< Coord3D > >& dfr )
{
    dfr.register_creator< DistanceDFCreator< Coord3D > >( "Distance" );
    dfr.register_creator< ConstantDFCreator< Coord3D > >( "Constant" );
}
}


#endif
