/*
 *  connection_functor.h
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

#ifndef CONNECTION_FUNCTOR_H
#define CONNECTION_FUNCTOR_H

#include <algorithm>

#include "numeric_functors.h"
#include "creator_registry.h"


namespace sapi
{
template < typename CoordT >
class ConnectionFunctor
{
public:
    ConnectionFunctor() = default;
    ConnectionFunctor( const ConnectionFunctor& );
    ConnectionFunctor( ConnectionFunctor&& );
    ~ConnectionFunctor() = default;

    ConnectionFunctor(
        const std::unique_ptr< DisplacementFunctor< CoordT > >&,
        const std::vector< std::unique_ptr< UnaryFunctor > >&
    );

    ConnectionFunctor(
        const std::string& df_name,
        const std::vector< space_t >& df_params,
        const CreatorRegistry< DisplacementFunctor< CoordT > >& df_reg,
        const std::vector< std::string >& ufs_names,
        const std::vector< std::vector< space_t > >& ufs_params,
        const CreatorRegistry< UnaryFunctor >& uf_reg
    );

    ConnectionFunctor& operator=( const ConnectionFunctor& );
    ConnectionFunctor& operator=( ConnectionFunctor&& );

    bool is_initialized() const;

    space_t operator()(
        const Displacement< CoordT >&
        ) const;

protected:
    void _clear();

    bool initialized_ = false;
    std::unique_ptr< DisplacementFunctor< CoordT > > df_;
    std::vector< std::unique_ptr< UnaryFunctor > > ufs_;
};


template < typename CoordT >
inline void ConnectionFunctor< CoordT >::_clear()
{
    df_.release();
    ufs_.clear();
    initialized_ = false;
}


template < typename CoordT >
ConnectionFunctor< CoordT >::ConnectionFunctor( const ConnectionFunctor& cf )
{
    if ( cf.initialized_ )
    {
        df_ = cf.df_->clone();
        ufs_.resize( cf.ufs_.size() );
        std::transform(
            cf.ufs_.cbegin(),
            cf.ufs_.cend(),
            ufs_.begin(),
            []( const auto& uf ) {
                return uf->clone();
            }
        );
        initialized_ = true;
    }
}


template < typename CoordT >
ConnectionFunctor< CoordT >::ConnectionFunctor( ConnectionFunctor&& cf )
{
    if ( cf.initialized_ )
    {
        df_ = std::move( cf.df_ );
        ufs_ = std::move( cf.ufs_ );
        initialized_ = true;

        cf._clear();
    }
}


template < typename CoordT >
ConnectionFunctor< CoordT >::ConnectionFunctor(
    const std::unique_ptr< DisplacementFunctor< CoordT > >& df,
    const std::vector< std::unique_ptr< UnaryFunctor > >& ufs
)
{
    if (
        !df ||
        std::any_of( ufs.cbegin(), ufs.cend(),
            []( const auto& uf ) { return !uf; } )
        )
        throw std::invalid_argument( "Cannot construct ConnectionFunctor using nullptrs" );

    df_ = df->clone();

    if ( !ufs.empty() )
    {
        ufs_.resize( ufs.size() );
        std::transform(
            ufs.cbegin(),
            ufs.cend(),
            ufs_.begin(),
            []( const auto& uf )
            { return uf->clone(); }
        );
    }

    initialized_ = true;
}


template < typename CoordT >
ConnectionFunctor< CoordT >::ConnectionFunctor(
    const std::string& df_name,
    const std::vector< space_t >& df_params,
    const CreatorRegistry< DisplacementFunctor< CoordT > >& df_reg,
    const std::vector< std::string >& ufs_names,
    const std::vector< std::vector< space_t > >& ufs_params,
    const CreatorRegistry< UnaryFunctor >& uf_reg
)
{
    if ( df_name.empty() )
    {
        if ( !( ufs_names.empty() && ufs_params.empty() ) )
            throw std::invalid_argument( "Cannot create ConnectionFunctor without DisplacementFunctor" );
        return;
    }

    df_ = df_reg.get_creator( df_name )->create( df_params );

    if ( ufs_names.size() != ufs_params.size() )
        throw std::invalid_argument( "Mismatched input vectors for UnaryFunctors given to CFGenerator" );

    if ( !ufs_names.empty() )
    {
        ufs_.resize( ufs_names.size() );
        std::transform(
            ufs_names.cbegin(),
            ufs_names.cend(),
            ufs_params.cbegin(),
            ufs_.begin(),
            [ & ]( const auto& uf_name, const auto& uf_params )
            { return uf_reg.get_creator( uf_name )->create( uf_params ); }
        );
    }

    initialized_ = true;
}


template < typename CoordT >
inline ConnectionFunctor< CoordT >&
ConnectionFunctor< CoordT >::operator=( const ConnectionFunctor& cf )
{
    _clear();

    if ( !cf.initialized_ )
        return *this;

    df_ = cf.df_->clone();

    if ( !cf.ufs_.empty() )
    {
        ufs_.resize( cf.ufs_.size() );
        std::transform(
            cf.ufs_.cbegin(),
            cf.ufs_.cend(),
            ufs_.begin(),
            []( const auto& uf ) {
                return uf->clone();
            }
        );
    }

    initialized_ = true;

    return *this;
}


template < typename CoordT >
inline ConnectionFunctor< CoordT >&
ConnectionFunctor< CoordT >::operator=( ConnectionFunctor&& cf )
{
    _clear();

    if ( !cf.initialized_ )
        return *this;

    df_ = std::move( cf.df_ );
    ufs_ = std::move( cf.ufs_ );
    initialized_ = true;

    cf._clear();

    return *this;
}


template < typename CoordT >
inline bool ConnectionFunctor< CoordT >::is_initialized() const
{
    return initialized_;
}


template < typename CoordT >
inline space_t ConnectionFunctor< CoordT >::operator()(
    const Displacement< CoordT >& dc
    ) const
{
    assert( initialized_ );
    auto val = ( *df_ )( dc );
    for ( const auto& uf : ufs_ )
        val = ( *uf )( val );

    return val;
}


template < typename CoordT >
struct CFCollection
{
    ConnectionFunctor< CoordT > weight_functor_;
    ConnectionFunctor< CoordT > delay_functor_;
    ConnectionFunctor< CoordT > probability_functor_;
};
}


#endif
