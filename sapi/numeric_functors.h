/*
 *  numeric_functors.h
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

#ifndef NUMERIC_FUNCTORS_H
#define NUMERIC_FUNCTORS_H

#include <cassert>

#include "numerics.h"
#include "type_erasure_helpers.h"


namespace sapi
{
// Forward definition to link with coordinates.h
template < typename CoordT >
struct Displacement;


struct UnaryFunctor : public Clonable< UnaryFunctor >
{
    virtual space_t operator()(
        const space_t& x
        ) const = 0;
};


template < typename UF >
struct CloningUFBase : public UnaryFunctor
{
    std::unique_ptr< UnaryFunctor > clone() const override
    {
        return std::make_unique< UF >( *dynamic_cast< const UF* >( this ) );
    }
};


struct IdentityUF : public CloningUFBase< IdentityUF >
{
    space_t operator()(
        const space_t& x
        ) const override
    {
        return x;
    }
};


struct MinUF : public CloningUFBase< MinUF >
{
    space_t min_ = 0;

    MinUF() = default;
    MinUF( const MinUF& ) = default;
    MinUF( MinUF&& ) = default;

    MinUF( const space_t& m )
        : min_( m )
    {
    }

    space_t operator()(
        const space_t& x
        ) const override
    {
        return std::fmin( x, min_ );
    }
};


struct MaxUF : public CloningUFBase< MaxUF >
{
    space_t max_ = 0;

    MaxUF() = default;
    MaxUF( const MaxUF& ) = default;
    MaxUF( MaxUF&& ) = default;

    MaxUF( const space_t& m )
        : max_( m )
    {
    }

    space_t operator()(
        const space_t& x
        ) const override
    {
        return std::fmax( x, max_ );
    }
};


struct InverseUF : public CloningUFBase< InverseUF >
{
    space_t operator()(
        const space_t& x
        ) const override
    {
        return almost_zero( x )
            ? std::signbit( x )
            ? -std::numeric_limits< space_t >::infinity()
            : std::numeric_limits< space_t >::infinity()
            : 1. / x;
    }
};


struct ProportionalUF : public CloningUFBase< ProportionalUF >
{
    space_t slope_ = 1;

    ProportionalUF() = default;
    ProportionalUF( const ProportionalUF& ) = default;
    ProportionalUF( ProportionalUF&& ) = default;

    ProportionalUF( const space_t& slope )
        : slope_( slope )
    {
    }

    space_t operator()(
        const space_t& x
        ) const override
    {
        return slope_ * x;
    }
};


struct UpperBoundUF : public CloningUFBase< UpperBoundUF >
{
    space_t threshold_ = 0.5;
    space_t clamp_ = 0;

    UpperBoundUF() = default;
    UpperBoundUF( const UpperBoundUF& ) = default;
    UpperBoundUF( UpperBoundUF&& ) = default;

    UpperBoundUF( const space_t& threshold, const space_t& clamp )
        : threshold_( threshold ), clamp_( clamp )
    {
    }

    space_t operator()(
        const space_t& x
        ) const override
    {
        return ( leq_test( threshold_, x ) ) ? clamp_ : x;
    }
};


struct LowerBoundUF : public CloningUFBase< LowerBoundUF >
{
    space_t threshold_ = 0.5;
    space_t clamp_ = 0;

    LowerBoundUF() = default;
    LowerBoundUF( const LowerBoundUF& ) = default;
    LowerBoundUF( LowerBoundUF&& ) = default;

    LowerBoundUF( const space_t& threshold, const space_t& clamp )
        : threshold_( threshold ), clamp_( clamp )
    {
    }

    space_t operator()(
        const space_t& x
        ) const override
    {
        return ( leq_test( x, threshold_ ) ) ? clamp_ : x;
    }
};


struct OffsetUF : public CloningUFBase< OffsetUF >
{
    space_t offset_ = 0;

    OffsetUF() = default;
    OffsetUF( const OffsetUF& ) = default;
    OffsetUF( OffsetUF&& ) = default;

    OffsetUF( const space_t& offset )
        : offset_( offset )
    {
    }

    space_t operator()(
        const space_t& x
        ) const override
    {
        return compensated_sum( offset_, x );
    }
};


struct ExponentialUF : public CloningUFBase< ExponentialUF >
{
    space_t neg_1_beta_ = -1;

    ExponentialUF() = default;
    ExponentialUF( const ExponentialUF& ) = default;
    ExponentialUF( ExponentialUF&& ) = default;

    ExponentialUF( const space_t& beta )
    {
        assert( !almost_zero( beta ) );
        neg_1_beta_ = -1. / beta;
    }

    space_t operator()(
        const space_t& x
        ) const override
    {
        return std::exp( x * neg_1_beta_ );
    }
};


struct GaussianUF : public CloningUFBase< GaussianUF >
{
    space_t neg_mean_ = 0;
    space_t neg_denominator_ = -1;

    GaussianUF() = default;
    GaussianUF( const GaussianUF& ) = default;
    GaussianUF( GaussianUF&& ) = default;

    GaussianUF( const space_t& mean, const space_t& std2 )
    {
        assert( !almost_zero( std2 ) );
        neg_mean_ = -mean;
        neg_denominator_ = -1. / ( 2. * std2 );
    }

    space_t operator()(
        const space_t& x
        ) const override
    {
        return std::exp( squared( compensated_sum( x, neg_mean_ ) ) * neg_denominator_ );
    }
};


template < typename CoordT >
struct DisplacementFunctor : public Clonable< DisplacementFunctor< CoordT > >
{
    virtual space_t operator()(
        const Displacement< CoordT >& dc
        ) const = 0;
};


template < typename CoordT, typename DFT >
struct CloningDFBase : public DisplacementFunctor< CoordT >
{
    std::unique_ptr< DisplacementFunctor< CoordT > > clone() const override
    {
        return std::make_unique< DFT >( *dynamic_cast< const DFT* >( this ) );
    }
};


template < typename CoordT >
struct DistanceDF : public CloningDFBase< CoordT, DistanceDF< CoordT > >
{
    space_t operator()(
        const Displacement< CoordT >& dc
        ) const override
    {
        return dc.get_distance();
    }
};


template < typename CoordT >
struct ConstantDF : public CloningDFBase< CoordT, ConstantDF< CoordT > >
{
    space_t c_ = 1;

    ConstantDF() = default;
    ConstantDF( const ConstantDF& ) = default;
    ConstantDF( ConstantDF&& ) = default;

    ConstantDF( const space_t& c )
        : c_( c )
    {
    }

    space_t operator()(
        const Displacement< CoordT >&
        ) const override
    {
        return c_;
    }
};
}


#endif
