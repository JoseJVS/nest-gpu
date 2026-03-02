/*
 *  tile.h
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

#ifndef TILE_H
#define TILE_H

#include <memory>
#include <string>
#include <vector>
#include <forward_list>
#include <cassert>

#include "coordinates.h"


// Forward definition to link with random_generators.h
namespace nest
{
    class BaseRandomGenerator;
    using RngPtr = BaseRandomGenerator*;
}


namespace sapi
{
template < typename CoordT >
class Tile
{
public:
    const tileidx_t index_;
    const CircumscribedRadius< CoordT > c_radius_;

    Tile() = delete;
    Tile( const Tile& tile ) = delete;
    Tile( Tile&& tile ) = default;
    virtual ~Tile() = default;

    Tile(
        const tileidx_t& index,
        const CoordT& origin,
        const space_t& radius
    );

    Tile(
        tileidx_t&& index,
        CircumscribedRadius< CoordT >&& c_radius,
        std::vector< CoordT >&& vertices
    );

    virtual void initialize_sub_tiles(
        const split_t& splits
    ) = 0;

    virtual bool coord_in_tile(
        const CoordT& coord
    ) const = 0;

    virtual tileidx_t compute_sub_tile_count(
        const split_t& splits
    ) const = 0;

    virtual CoordT project_point_to_surface(
        const CoordT& coord
    ) const = 0;

    virtual std::vector< split_t >
        get_possible_sub_tile_branches(
            const split_t& known_split_order
        ) const = 0;

    virtual tileidx_t
        compute_leaf_index(
            const std::vector< split_t >& branch_sequence
        ) const = 0;

    virtual void generate_coords_in_tile(
        std::vector< CoordT >& coord_vec,
        nest::RngPtr const& rng
    ) const = 0;

    virtual std::string get_name() const = 0;

    bool is_split() const;

    const std::vector< CoordT >&
        get_vertices() const;

    const std::vector< std::unique_ptr< Tile > >&
        get_sub_tiles() const;

    void insert_leaf_sub_tiles(
        std::forward_list< const Tile* >& lst_container
    ) const;

    std::vector< const Tile* > get_leaf_sub_tiles(
        const split_t& known_split_order
    ) const;

    std::vector< std::vector< space_t > >
        export_vertices_to_nested_vec() const;

    std::string to_string( const uint8_t& tabs = 0 ) const;

    bool operator==( const Tile& ) const;

protected:
    std::vector< CoordT > vertices_;
    std::vector< std::unique_ptr< Tile > > sub_tiles_;

    template < typename IT >
    friend void iterate_sub_tiles_depth_first(
        IT& it,
        const Tile< CoordT >* const& sub_tile
    )
    {
        if ( !sub_tile->sub_tiles_.empty() )
            for ( const auto& st : sub_tile->sub_tiles_ )
                iterate_sub_tiles_depth_first( it, st.get() );
        else
            *it++ = sub_tile;
    }
};


template < typename CoordT >
Tile< CoordT >::Tile(
    const tileidx_t& index,
    const CoordT& origin,
    const space_t& radius
)
    : index_( index )
    , c_radius_(
        CircumscribedRadius< CoordT >( origin, squared( radius ) )
    )
{
    assert(
        0 <= index &&
        !almost_zero( c_radius_.radius2_ ) &&
        !std::signbit( radius )
    );
}


template < typename CoordT >
Tile< CoordT >::Tile(
    tileidx_t&& index,
    CircumscribedRadius< CoordT >&& c_radius,
    std::vector< CoordT >&& vertices
)
    : index_( std::move( index ) )
    , c_radius_( std::move( c_radius ) )
    , vertices_( std::move( vertices ) )
{
    assert(
        0 <= index_ &&
        !almost_zero( c_radius_.radius2_ )
    );
}


template < typename CoordT >
inline bool Tile< CoordT >::is_split() const
{
    return !sub_tiles_.empty();
}


template < typename CoordT >
inline const std::vector< CoordT >&
Tile< CoordT >::get_vertices() const
{
    return vertices_;
}


template < typename CoordT >
inline const std::vector< std::unique_ptr< Tile< CoordT > > >&
Tile< CoordT >::get_sub_tiles() const
{
    return sub_tiles_;
}


template < typename CoordT >
void Tile< CoordT >::insert_leaf_sub_tiles(
    std::forward_list< const Tile* >& lst_container
) const
{
    std::forward_list< const Tile* > tree{ this };
    do
    {
        auto tile = tree.front();
        tree.pop_front();

        if ( tile->is_split() )
            for ( const auto& st : tile->sub_tiles_ )
                tree.emplace_front( st.get() );
        else
            lst_container.emplace_front( std::move( tile ) );

    } while ( !tree.empty() );
}


template < typename CoordT >
std::vector< const Tile< CoordT >* >
Tile< CoordT >::get_leaf_sub_tiles(
    const split_t& known_split_order
) const
{
    if ( known_split_order == 0 )
        return std::vector< const Tile* >{ this };

    const auto leaf_count = compute_sub_tile_count( known_split_order );
    assert( leaf_count > 1 );

    std::vector < const Tile* > lst_vec( leaf_count, nullptr );
    auto lst_it = lst_vec.begin();

    iterate_sub_tiles_depth_first(
        lst_it,
        this
    );

    assert( lst_it == lst_vec.end() );

    return lst_vec;
}


template < typename CoordT >
std::vector< std::vector< space_t > >
Tile< CoordT >::export_vertices_to_nested_vec() const
{
    std::vector< std::vector< space_t > >
        nested_spaceT_vec( vertices_.size() + 1 );
    auto nv_it = nested_spaceT_vec.begin();

    nv_it->resize( CoordT::D );
    c_radius_.origin_.copy_to_vec( ( *nv_it++ ).begin() );
    for ( const auto& vertex : vertices_ )
    {
        nv_it->resize( CoordT::D );
        vertex.copy_to_vec( ( *nv_it++ ).begin() );
    }

    return nested_spaceT_vec;
}


template < typename CoordT >
std::string Tile< CoordT >::to_string( const uint8_t& tabs ) const
{
    std::string str = "";
    for ( uint8_t t = 0; t < tabs; ++t )
        str += "\t";

    str += get_name()
        + " - index: "
        + std::to_string( index_ )
        + ", properties: "
        + c_radius_.to_string()
        + " , vertices: ";

    for ( const auto& vt : vertices_ )
        str += vt.to_string();

    for ( const auto& st : sub_tiles_ )
        str += "\n" + st->to_string( tabs + 1 );

    return str;
}


template < typename CoordT >
inline bool  Tile< CoordT >::operator==( const Tile& t ) const
{
    // The circumscribed radius and the vertex vectors
    // are enough to uniquely identify a tile
    // sub tiles are generated from the given properties of a tile
    return c_radius_ == t.c_radius_ && vertices_ == t.vertices_;
}
}


#endif
