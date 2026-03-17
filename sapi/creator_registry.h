/*
 *  creator_registry.h
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

#ifndef CREATOR_REGISTRY_H
#define CREATOR_REGISTRY_H

#include <string>
#include <unordered_map>

#include "type_erasure_helpers.h"


namespace sapi
{
template < typename RT >
class CreatorRegistry
{
public:
    CreatorRegistry() = default;
    CreatorRegistry( const CreatorRegistry& ) = delete;
    CreatorRegistry( CreatorRegistry&& ) = default;
    ~CreatorRegistry() = default;

    template < typename CT,
        typename std::enable_if_t<
        std::is_base_of_v< StateLessCreator< RT >, CT >,
        bool > = true >
    void register_creator( std::string&& );

    const std::unique_ptr< StateLessCreator< RT > >&
        get_creator( const std::string& ) const;

protected:
    std::unordered_map< std::string,
        std::unique_ptr< StateLessCreator< RT > > > registry_;
};


template < typename RT >
template < typename CT,
    typename std::enable_if_t<
    std::is_base_of_v< StateLessCreator< RT >, CT >,
    bool > >
inline void CreatorRegistry< RT >::register_creator(
    std::string&& name
)
{
    if ( registry_.find( name ) != registry_.end() )
        throw std::invalid_argument( name + " is already registered" );
    else
        registry_.emplace(
            std::make_pair(
                std::move( name ),
                std::make_unique< CT >()
            )
        );
}


template < typename RT >
inline const std::unique_ptr< StateLessCreator< RT > >&
CreatorRegistry< RT >::get_creator(
    const std::string& name
) const
{
    const auto search = registry_.find( name );
    if ( search == registry_.end() )
        throw std::invalid_argument( name + " not known in registry" );
    return search->second;
}
}


#endif
