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

#include <forward_list>
#include <cassert>

#include "coordinates.h"
#include "tile_geometry.h"


namespace sapi
{
template < typename CoordT >
struct Tile
{
    TILE_SHAPE shape_ = TILE_SHAPE::NULL_TS;
    tileidx_t index_ = 0;
    CircumscribedRadius< CoordT > c_radius_;
    std::vector< CoordT > vertices_;
    std::vector< CoordT > helper_vectors_;
    std::vector< space_t > helper_scalars_;
    mutable std::vector< Tile > sub_tiles_;

    Tile() = default;
    Tile( const Tile& ) = delete;
    Tile( Tile&& ) = default;
    ~Tile() = default;

    Tile(
        const TILE_SHAPE&,
        const CoordT&,
        const std::vector< space_t >&,
        const std::vector< angle_t >&
    );

    Tile(
        TILE_SHAPE&&,
        tileidx_t&&,
        CircumscribedRadius< CoordT >&&,
        std::vector< CoordT >&&
    );

    Tile& operator=( Tile&& );

    void split_tile(
        const split_t&
    ) const;

    bool coord_in_tile(
        const CoordT&
    ) const;

    void generate_coords_in_tile(
        std::vector< CoordT >&,
        AnyRNG&
    ) const;

    CoordT project_point_to_surface(
        const CoordT&
    ) const;

    tileidx_t compute_sub_tile_count(
        const split_t&
    ) const;

    std::vector< split_t >
        get_possible_sub_tile_branches(
            const split_t&
        ) const;

    void insert_leaf_sub_tiles(
        std::forward_list< const Tile* >&
    ) const;

    std::vector< const Tile* > get_leaf_sub_tiles(
        const split_t&
    ) const;

    std::vector< std::vector< space_t > >
        export_vertices_to_nested_vec() const;

    bool operator==( const Tile& ) const;

    std::string get_name() const;

    std::string to_string( const uint8_t& = 0 ) const;

    template < typename IT >
    friend void iterate_sub_tiles_depth_first(
        IT& it,
        const Tile< CoordT >* const& sub_tile
    )
    {
        if ( !sub_tile->sub_tiles_.empty() )
            for ( const auto& st : sub_tile->sub_tiles_ )
                iterate_sub_tiles_depth_first( it, &st );
        else
            *it++ = sub_tile;
    }
};


template < typename CoordT >
Tile< CoordT >::Tile(
    const TILE_SHAPE& shape,
    const CoordT& origin,
    const std::vector< space_t >& side_lengths,
    const std::vector< angle_t >& angular_offsets
)
    : shape_( shape )
{
    if ( side_lengths.empty() )
        throw std::invalid_argument( "Invalid tile side lengths vector" );

    c_radius_.origin_ = origin;
    initialize_tile_vertices(
        vertices_,
        c_radius_,
        side_lengths,
        angular_offsets,
        shape_
    );
    initialize_tile_helpers(
        helper_vectors_,
        helper_scalars_,
        vertices_,
        shape_
    );
}


template < typename CoordT >
Tile< CoordT >::Tile(
    TILE_SHAPE&& shape,
    tileidx_t&& index,
    CircumscribedRadius< CoordT >&& c_radius,
    std::vector< CoordT >&& vertices
)
    : shape_( shape )
    , index_( index )
    , c_radius_( std::move( c_radius ) )
    , vertices_( std::move( vertices ) )
{
    assert(
        0 <= index_ &&
        !almost_zero( c_radius_.radius2_ )
    );
}


template < typename CoordT >
inline Tile< CoordT >&
Tile< CoordT >::operator=( Tile&& t )
{
    shape_ = t.shape_;
    index_ = t.index_;
    c_radius_ = std::move( t.c_radius_ );
    vertices_ = std::move( t.vertices_ );
    helper_vectors_ = std::move( t.helper_vectors_ );
    helper_scalars_ = std::move( t.helper_scalars_ );
    sub_tiles_ = std::move( t.sub_tiles_ );

    t.shape_ = TILE_SHAPE::NULL_TS;

    return *this;
}


template < typename CoordT >
inline void Tile< CoordT >::split_tile( const split_t& splits ) const
{
    sapi::split_tile(
        sub_tiles_,
        vertices_,
        c_radius_,
        index_,
        splits,
        shape_
    );
}


template < typename CoordT >
inline bool Tile< CoordT >::coord_in_tile( const CoordT& coord ) const
{
    return sapi::coord_in_tile(
        coord,
        c_radius_,
        vertices_,
        helper_vectors_,
        helper_scalars_,
        shape_
    );
}


template < typename CoordT >
inline void Tile< CoordT >::generate_coords_in_tile(
    std::vector< CoordT >& coord_vec,
    AnyRNG& rng
) const
{
    sapi::generate_coords_in_tile(
        coord_vec,
        rng,
        vertices_,
        helper_vectors_,
        c_radius_,
        shape_
    );
}


template < typename CoordT >
inline CoordT Tile< CoordT >::project_point_to_surface(
    const CoordT& coord
) const
{
    return sapi::project_point_to_surface( coord, vertices_ );
}


template < typename CoordT >
inline tileidx_t Tile< CoordT >::compute_sub_tile_count(
    const split_t& splits
) const
{
    return sapi::compute_sub_tile_count( splits, shape_ );
}


template < typename CoordT >
inline std::vector< split_t >
Tile< CoordT >::get_possible_sub_tile_branches(
    const split_t& splits
) const
{
    return sapi::get_possible_sub_tile_branches( splits, shape_ );
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

        if ( !tile->sub_tiles_.empty() )
            for ( const auto& st : tile->sub_tiles_ )
                tree.emplace_front( &st );
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
inline std::string Tile< CoordT >::get_name() const
{
    return sapi::get_name( shape_ );
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
        str += "\n" + st.to_string( tabs + 1 );

    return str;
}


template < typename CoordT >
inline bool  Tile< CoordT >::operator==( const Tile& t ) const
{
    // The circumscribed radius and the vertex vectors
    // are enough to uniquely identify a tile
    // sub tiles are generated from the given properties of a tile
    return shape_ == t.shape_ && c_radius_ == t.c_radius_ && vertices_ == t.vertices_;
}
}


#endif
