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

#include <vector>
#include <stdexcept>
#include <cassert>

#include "numerics.h"
#include "enum_store.h"


namespace sapi
{
// Forward definition to link with coordinates.h
struct Coord2D;
struct Coord3D;
template < typename CoordT >
struct Displacement;

// Forward definition to link with creator_registry.h
template < typename T >
class CreatorRegistry;


struct UnaryFunctor
{
    UNARY_FUNCTION func_ = UNARY_FUNCTION::NULL_UF;
    space_t var_[ 2 ] = { 0. };

    UnaryFunctor() = default;
    UnaryFunctor( const UnaryFunctor& );
    UnaryFunctor( UnaryFunctor&& ) = default;
    ~UnaryFunctor() = default;

    UnaryFunctor(
        const UNARY_FUNCTION&,
        const space_t & = 0,
        const space_t & = 0
    );

    UnaryFunctor& operator=( const UnaryFunctor& );

    inline space_t operator()( const space_t& ) const;
};


inline UnaryFunctor&
UnaryFunctor::operator=( const UnaryFunctor& uf )
{
    func_ = uf.func_;
    var_[ 0 ] = uf.var_[ 0 ];
    var_[ 1 ] = uf.var_[ 1 ];
    return *this;
}


inline space_t
UnaryFunctor::operator()( const space_t& x ) const
{
    switch ( func_ )
    {
    case UNARY_FUNCTION::IDENTITY:
        return x;

    case UNARY_FUNCTION::MIN:
        return std::fmin( x, var_[ 0 ] );

    case UNARY_FUNCTION::MAX:
        return std::fmax( x, var_[ 0 ] );

    case UNARY_FUNCTION::LOWER_BOUND:
        return leq_test( x, var_[ 0 ] ) ? var_[ 1 ] : x;

    case UNARY_FUNCTION::UPPER_BOUND:
        return leq_test( var_[ 0 ], x ) ? var_[ 1 ] : x;

    case UNARY_FUNCTION::INVERSE:
        return almost_zero( x )
            ? std::signbit( x )
            ? -std::numeric_limits< space_t >::infinity()
            : std::numeric_limits< space_t >::infinity()
            : 1. / x;

    case UNARY_FUNCTION::FACTOR:
        return x * var_[ 0 ];

    case UNARY_FUNCTION::OFFSET:
        return compensated_sum( x, var_[ 0 ] );

    case UNARY_FUNCTION::EXPONENTIAL:
        return std::exp( x * var_[ 0 ] );

    case UNARY_FUNCTION::GAUSSIAN:
        return std::exp( squared( compensated_sum( x, var_[ 0 ] ) ) * var_[ 1 ] );

    default:
        throw std::invalid_argument( "Invalid unary function name" );
    }
}


struct DisplacementFunctor
{
    DISPLACEMENT_FUNCTION func_ = DISPLACEMENT_FUNCTION::NULL_DF;
    space_t var_ = 0;

    DisplacementFunctor() = default;
    DisplacementFunctor( const DisplacementFunctor& ) = default;
    DisplacementFunctor( DisplacementFunctor&& ) = default;
    ~DisplacementFunctor() = default;

    DisplacementFunctor(
        const DISPLACEMENT_FUNCTION&,
        const space_t & = 0
    );

    DisplacementFunctor& operator=( const DisplacementFunctor& );

    template < typename CoordT >
    space_t operator()( const Displacement< CoordT >& ) const;
};


inline DisplacementFunctor&
DisplacementFunctor::operator=( const DisplacementFunctor& df )
{
    func_ = df.func_;
    var_ = df.var_;
    return *this;
}


template < typename CoordT >
inline space_t
DisplacementFunctor::operator()( const Displacement< CoordT >& d ) const
{
    switch ( func_ )
    {
    case DISPLACEMENT_FUNCTION::CONSTANT:
        return var_;

    case DISPLACEMENT_FUNCTION::DISTANCE:
        return d.get_distance();

    case DISPLACEMENT_FUNCTION::DISPLACEMENT_X:
        return d.displacement_.x_;

    case DISPLACEMENT_FUNCTION::DISPLACEMENT_Y:
        return d.displacement_.y_;

    case DISPLACEMENT_FUNCTION::DISTANCE_X:
        return std::fabs( d.displacement_.x_ );

    case DISPLACEMENT_FUNCTION::DISTANCE_Y:
        return std::fabs( d.displacement_.y_ );

    default:
        break;
    }

    if constexpr ( std::is_same_v< CoordT, Coord3D > )
    {
        switch ( func_ )
        {
        case DISPLACEMENT_FUNCTION::DISPLACEMENT_Z:
            return d.displacement_.z_;

        case DISPLACEMENT_FUNCTION::DISTANCE_Z:
            return std::fabs( d.displacement_.z_ );

        default:
            break;
        }
    }

    throw std::invalid_argument( "Invalid displacement function name" );
}


class NumericFunctor
{
public:
    NumericFunctor() = default;
    NumericFunctor( const NumericFunctor& ) = default;
    NumericFunctor( NumericFunctor&& ) = default;
    ~NumericFunctor() = default;

    NumericFunctor(
        const DisplacementFunctor&,
        const std::vector< UnaryFunctor >&
    );

    NumericFunctor(
        const std::string&,
        const std::vector< space_t >&,
        const CreatorRegistry< DisplacementFunctor >&,
        const std::vector< std::string >&,
        const std::vector< std::vector< space_t > >&,
        const CreatorRegistry< UnaryFunctor >&
    );

    NumericFunctor& operator=( const NumericFunctor& );
    NumericFunctor& operator=( NumericFunctor&& );

    bool is_initialized() const;

    template < typename CoordT >
    space_t operator()(
        const Displacement< CoordT >&
        ) const;

protected:
    void _clear();

    bool initialized_ = false;
    DisplacementFunctor df_;
    std::vector< UnaryFunctor > ufs_;
};


inline void NumericFunctor::_clear()
{
    df_.func_ = DISPLACEMENT_FUNCTION::NULL_DF;
    ufs_.clear();
    initialized_ = false;
}


inline NumericFunctor&
NumericFunctor::operator=( const NumericFunctor& cf )
{
    _clear();

    if ( !cf.initialized_ )
        return *this;

    df_ = cf.df_;
    ufs_ = cf.ufs_;

    initialized_ = true;

    return *this;
}


inline NumericFunctor&
NumericFunctor::operator=( NumericFunctor&& cf )
{
    _clear();

    if ( !cf.initialized_ )
        return *this;

    df_ = cf.df_;
    ufs_ = std::move( cf.ufs_ );
    initialized_ = true;

    cf._clear();

    return *this;
}


inline bool NumericFunctor::is_initialized() const
{
    return initialized_;
}


template < typename CoordT >
inline space_t NumericFunctor::operator()(
    const Displacement< CoordT >& dc
    ) const
{
    assert( initialized_ );
    auto val = df_( dc );
    for ( const auto& uf : ufs_ )
        val = uf( val );

    return val;
}


struct NFCollection
{
    NumericFunctor weight_functor_;
    NumericFunctor delay_functor_;
    NumericFunctor probability_functor_;
};
}


#endif
