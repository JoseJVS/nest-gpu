/*
 *  garbage_collector.h
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

#ifndef GARBAGE_COLLECTOR_H
#define GARBAGE_COLLECTOR_H

#include <memory>
#include <forward_list>


namespace sapi
{
struct AnyPtr
{
    struct PtrConcept
    {
    };

    template < typename T >
    struct PtrModel : public PtrConcept
    {
        std::unique_ptr< T > ptr_;

        PtrModel( std::unique_ptr< T >&& ptr ) : ptr_( std::move( ptr ) )
        {
        }

        PtrModel() = delete;
        PtrModel( const PtrModel& ) = delete;
        PtrModel( PtrModel&& ) = default;
        ~PtrModel() = default;
    };

    std::unique_ptr< PtrConcept > internal_ptr_;

    template < typename T >
    AnyPtr( std::unique_ptr< T >&& ap )
        : internal_ptr_(
            std::make_unique< PtrModel< T > >( std::move( ap ) )
        )
    {
    }

    AnyPtr() = delete;
    AnyPtr( const AnyPtr& ) = delete;
    AnyPtr( AnyPtr&& ) = default;
    ~AnyPtr() = default;
};


struct GC
{
    std::forward_list< AnyPtr > gc_fl_;

    GC() = default;
    GC( const GC& ) = delete;
    GC( GC&& ) = default;
    ~GC() = default;

    template < typename T >
    T* make_collected()
    {
        auto u_ptr = std::make_unique< T >();
        auto ptr = u_ptr.get();
        gc_fl_.emplace_front( std::move( u_ptr ) );
        return ptr;
    }

    template < typename T >
    T* make_collected( const std::size_t& size )
    {
        if ( size < 1 )
            return static_cast< T* >( nullptr );

        auto u_ptr = std::make_unique< T[] >( size );
        auto ptr = u_ptr.get();
        gc_fl_.emplace_front( std::move( u_ptr ) );
        return ptr;
    }

    template < typename T >
    void collect( std::unique_ptr< T >&& ptr )
    {
        gc_fl_.emplace_front( std::move( ptr ) );
    }

    void free_gc()
    {
        gc_fl_.clear();
    }
};
}


#endif
