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
#include <stdexcept>

#include "sapi_config.h"


namespace sapi
{
template < typename T >
struct Clonable
{
    virtual ~Clonable() = default;
    virtual std::unique_ptr< T > clone() const = 0;
};


template < typename RT >
struct StateLessCreator
{
    virtual ~StateLessCreator() = default;

    virtual std::unique_ptr< RT > create() const
    {
        throw std::runtime_error( "Undefined creation method for state-less creator" );
    }

    virtual std::unique_ptr< RT > create( const std::vector< space_t >& ) const
    {
        throw std::runtime_error( "Undefined creation method for state-less creator" );
    }

    virtual std::unique_ptr< RT > create(
        const std::vector< space_t >&,
        const std::vector< space_t >&
    ) const
    {
        throw std::runtime_error( "Undefined creation method for state-less creator" );
    }
};
}


#endif
