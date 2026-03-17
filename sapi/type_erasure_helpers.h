/*
 *  type_erasure_helpers.h
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

#ifndef TYPE_ERASURE_HELPERS_H
#define TYPE_ERASURE_HELPERS_H

#include <memory>
#include <vector>
#include <limits>
#include <utility>
#include <stdexcept>
#include <type_traits>
#include <initializer_list>

#include "sapi_config.h"


namespace sapi
{
template < typename T >
struct Cloneable
{
    virtual ~Cloneable() = default;
    virtual std::unique_ptr< T > clone() const = 0;
};


template < typename RT >
struct StateLessCreator
{
    virtual ~StateLessCreator() = default;

    virtual RT create() const;

    virtual RT create( const std::vector< space_t >& ) const;

    virtual RT create(
        const std::vector< space_t >&,
        const std::vector< angle_t >&
    ) const;

    virtual RT create(
        const std::vector< space_t >&,
        const std::vector< space_t >&,
        const std::vector< space_t >&
    ) const;
};


template < typename RT >
inline RT StateLessCreator< RT >::create() const
{
    throw std::runtime_error( "Undefined creation method for state-less creator" );
}


template < typename RT >
inline RT StateLessCreator< RT >::create( const std::vector< space_t >& ) const
{
    throw std::runtime_error( "Undefined creation method for state-less creator" );
}


template < typename RT >
inline RT StateLessCreator< RT >::create(
    const std::vector< space_t >&,
    const std::vector< angle_t >&
) const
{
    throw std::runtime_error( "Undefined creation method for state-less creator" );
}


template < typename RT >
inline RT StateLessCreator< RT >::create(
    const std::vector< space_t >&,
    const std::vector< space_t >&,
    const std::vector< space_t >&
) const
{
    throw std::runtime_error( "Undefined creation method for state-less creator" );
}


template < typename RT >
struct RNGConcept : public Cloneable< RNGConcept< RT > >
{
    virtual void seed( const std::initializer_list< uint32_t >& ) = 0;
    virtual RT operator()() = 0;
};


template < typename RT, typename RNG >
struct RNGModel final : public RNGConcept< RT >
{
    RNG engine_;

    RNGModel( RNG );

    // Defined in random_manager.cpp
    void seed( const std::initializer_list< uint32_t >& ) override;

    RT operator()() override;

    std::unique_ptr< RNGConcept< RT > > clone() const override;
};


template < typename RT, typename RNG >
RNGModel< RT, RNG >::RNGModel( RNG engine )
    : engine_( engine )
{
}


template < typename RT, typename RNG >
inline RT RNGModel< RT, RNG >::operator()()
{
    return engine_();
}


template < typename RT, typename RNG >
inline std::unique_ptr< RNGConcept< RT > >
RNGModel< RT, RNG >::clone() const
{
    return std::make_unique< RNGModel >( engine_ );
}


template < typename RT,
    typename std::enable_if_t<
    std::disjunction_v<
    std::is_same< RT, uint32_t >,
    std::is_same< RT, uint64_t >
    >
    , bool > = true
>
class AnyRNG_T final : public Cloneable < AnyRNG_T< RT > >
{
public:
    using result_type = RT;

    AnyRNG_T() = delete;
    AnyRNG_T( const AnyRNG_T& ) = delete;
    AnyRNG_T( AnyRNG_T&& ) = default;

    template < typename RNG >
    AnyRNG_T( RNG );

    AnyRNG_T( std::unique_ptr< RNGConcept< RT > >&& );

    constexpr static RT min();

    constexpr static RT max();

    void seed( const std::initializer_list< uint32_t >& );

    RT operator()();

    std::unique_ptr< AnyRNG_T > clone() const override;

protected:
    std::unique_ptr< RNGConcept< RT > > rng_;
    RNGConcept< RT >* ptr_;
};


template < typename RT,
    typename std::enable_if_t<
    std::disjunction_v<
    std::is_same< RT, uint32_t >,
    std::is_same< RT, uint64_t >
    >
    , bool > b
>
template < typename RNG >
AnyRNG_T< RT, b >::AnyRNG_T( RNG engine )
    : rng_( std::make_unique< RNGModel< RT, RNG > >( engine ) )
    , ptr_( rng_.get() )
{
}


template < typename RT,
    typename std::enable_if_t<
    std::disjunction_v<
    std::is_same< RT, uint32_t >,
    std::is_same< RT, uint64_t >
    >
    , bool > b
>
AnyRNG_T< RT, b >::AnyRNG_T( std::unique_ptr< RNGConcept< RT > >&& ptr )
    : rng_( std::move( ptr ) )
    , ptr_( rng_.get() )
{
}


template < typename RT,
    typename std::enable_if_t<
    std::disjunction_v<
    std::is_same< RT, uint32_t >,
    std::is_same< RT, uint64_t >
    >
    , bool > b
>
constexpr RT AnyRNG_T< RT, b >::min()
{
    return 0;
}


template < typename RT,
    typename std::enable_if_t<
    std::disjunction_v<
    std::is_same< RT, uint32_t >,
    std::is_same< RT, uint64_t >
    >
    , bool > b
>
constexpr RT AnyRNG_T< RT, b >::max()
{
    return std::numeric_limits< RT >::max();
}


template < typename RT,
    typename std::enable_if_t<
    std::disjunction_v<
    std::is_same< RT, uint32_t >,
    std::is_same< RT, uint64_t >
    >
    , bool > b
>
inline void AnyRNG_T< RT, b >::seed(
    const std::initializer_list< uint32_t >& l
)
{
    ptr_->seed( l );
}


template < typename RT,
    typename std::enable_if_t<
    std::disjunction_v<
    std::is_same< RT, uint32_t >,
    std::is_same< RT, uint64_t >
    >
    , bool > b
>
inline RT AnyRNG_T< RT, b >::operator()()
{
    return ptr_->operator()();
}


template < typename RT,
    typename std::enable_if_t<
    std::disjunction_v<
    std::is_same< RT, uint32_t >,
    std::is_same< RT, uint64_t >
    >
    , bool > b
>
inline std::unique_ptr< AnyRNG_T< RT, b > >
AnyRNG_T< RT, b >::clone() const
{
    return std::make_unique< AnyRNG_T >( ptr_->clone() );
}


typedef AnyRNG_T< rng_bits_t > AnyRNG;
}


#endif
