/*
 *  tile_grid.h
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

#ifndef TILE_GRID_H
#define TILE_GRID_H

#include <set>
#include <string>
#include <utility>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include "grid_containers.h"


namespace sapi
{
// Forward definition to tile.h
template < typename CoordT >
struct Tile;

// Forward definition to bounding_box.h
template < typename CoordT >
struct BoundingBox;


template < typename CoordT >
struct ShiftedImage
{
    GridPosition< CoordT > shifted_position_;
    std::optional< Tile< CoordT > > shifted_tile_;
    // Create shifted displacement to compute distance between nodes.
    // Node shifts can be computed relative to the origin,
    // hence we the displacement from the origin to the shifted
    // origin to each node to get its position relative to the shifted origin
    std::optional< CoordT > shift_displacement_;

    ShiftedImage() = default;
    ShiftedImage( const ShiftedImage& ) = delete;
    ShiftedImage( ShiftedImage&& ) = default;
    ~ShiftedImage() = default;

    ShiftedImage& operator=( ShiftedImage&& );

    bool operator==( const ShiftedImage& ) const;
};


template < typename CoordT >
inline ShiftedImage< CoordT >&
ShiftedImage< CoordT >::operator=( ShiftedImage&& si )
{
    shifted_position_ = std::move( si.shifted_position_ );
    shifted_tile_ = std::move( si.shifted_tile_ );
    shift_displacement_ = std::move( si.shift_displacement_ );

    return *this;
}


template < typename CoordT >
inline bool ShiftedImage< CoordT >::operator==( const ShiftedImage& si ) const
{
    return shifted_position_ == si.shifted_position_ &&
        shift_displacement_ == si.shift_displacement_ &&
        shifted_tile_ == si.shifted_tile_;
}


template < typename CoordT >
struct TilePosition
{
    GridPosition< CoordT > position_;
    Tile< CoordT > tile_;
    std::vector< ShiftedImage< CoordT > > grid_images_;
    std::unordered_set< tileidx_t > direct_tile_neighborhood_;
    std::unordered_set< tileidx_t > wrapped_tile_neighborhood_;

    TilePosition() = default;
    TilePosition( const TilePosition& ) = delete;
    TilePosition( TilePosition&& ) = default;
    ~TilePosition() = default;

    TilePosition& operator=( TilePosition&& );

    const std::unordered_set< tileidx_t >&
        get_tile_neighborhood( const bool& ) const;

    bool in_neighborhood(
        const tileidx_t&, const bool&
    ) const;

    std::string to_string() const;

    bool operator==( const TilePosition& ) const;
};


template < typename CoordT >
inline TilePosition< CoordT >&
TilePosition< CoordT >::operator=( TilePosition&& tp )
{
    position_ = std::move( tp.position_ );
    tile_ = std::move( tp.tile_ );
    grid_images_ = std::move( tp.grid_images_ );
    direct_tile_neighborhood_ = std::move( tp.direct_tile_neighborhood_ );
    wrapped_tile_neighborhood_ = std::move( tp.wrapped_tile_neighborhood_ );

    return *this;
}


template < typename CoordT >
inline const std::unordered_set< tileidx_t >&
TilePosition< CoordT >::get_tile_neighborhood( const bool& edge_wrap ) const
{
    return edge_wrap ? wrapped_tile_neighborhood_ : direct_tile_neighborhood_;
}


template < typename CoordT >
inline bool TilePosition< CoordT >::in_neighborhood(
    const tileidx_t& index, const bool& edge_wrap
) const
{
    if ( edge_wrap )
        return wrapped_tile_neighborhood_.find( index ) != wrapped_tile_neighborhood_.end();
    else
        return direct_tile_neighborhood_.find( index ) != direct_tile_neighborhood_.end();
}


template < typename CoordT >
std::string TilePosition< CoordT >::to_string() const
{
    std::string res = tile_->c_radius_.origin_.to_string() + ": [ ";
    auto num_neighbors = wrapped_tile_neighborhood_.size();
    for ( const auto& neighbor_index : wrapped_tile_neighborhood_ )
        res += std::to_string( neighbor_index ) +
        ( num_neighbors-- > 1 ? ", " : " ]" );

    return res;
}


template < typename CoordT >
inline bool TilePosition< CoordT >::operator==( const TilePosition& tp ) const
{
    return position_ == tp.position_ &&
        direct_tile_neighborhood_ == tp.direct_tile_neighborhood_ &&
        wrapped_tile_neighborhood_ == tp.wrapped_tile_neighborhood_ &&
        tile_ == tp.tile_ &&
        grid_images_ == tp.grid_images_;
}


template < typename CoordT >
struct TileGrid
{
    split_t splits_ = 0;
    bool has_split_ = false;
    tileidx_t num_tiles_ = 0;
    GridPosition< CoordT > dimensions_;
    BoundingBox< CoordT > bounding_box_;
    std::vector< TilePosition< CoordT > > positions_;

    TileGrid() = default;
    TileGrid( const TileGrid& ) = delete;
    TileGrid( TileGrid&& ) = default;
    ~TileGrid() = default;

    TileGrid( const GridPosition < CoordT >& );

    TileGrid& operator=( TileGrid&& );

    void split_owned_tiles(
        const split_t&,
        const std::set< tileidx_t >&
    );

    std::string to_string() const;

    bool operator==( const TileGrid& ) const;
};


template < typename CoordT >
TileGrid< CoordT >::TileGrid( const GridPosition< CoordT >& dimensions )
    : num_tiles_( 1 )
    , dimensions_( dimensions )
{
    for ( const auto& d : dimensions_ )
        num_tiles_ *= d;
    if ( num_tiles_ < 1 )
        throw std::invalid_argument( "Invalid grid dimensions" );
    positions_.resize( num_tiles_ );
}


template < typename CoordT >
inline TileGrid< CoordT >&
TileGrid< CoordT >::operator=( TileGrid&& tg )
{
    splits_ = tg.splits_;
    tg.splits_ = 0;
    has_split_ = tg.has_split_;
    tg.has_split_ = false;
    num_tiles_ = tg.num_tiles_;
    tg.num_tiles_ = 0;

    dimensions_ = std::move( tg.dimensions_ );
    bounding_box_ = std::move( tg.bounding_box_ );
    positions_ = std::move( tg.positions_ );

    return *this;
}


template < typename CoordT >
void TileGrid< CoordT >::split_owned_tiles(
    const split_t& num_splits,
    const std::set< tileidx_t >& owned_tiles
)
{
    assert( !positions_.empty() && !has_split_ );
    if ( num_splits < 0 )
        throw std::invalid_argument( "Cannot negatively split" );

    if ( owned_tiles.empty() )
    {
        has_split_ = true;
        return;
    }

#pragma omp parallel default( none )\
shared( positions_, owned_tiles, num_splits )
#pragma omp master
#pragma omp taskgroup
    for ( const auto& position : owned_tiles )
        positions_.at( position ).tile_.split_tile( num_splits );

    splits_ = num_splits;
    has_split_ = true;
}


template < typename CoordT >
std::string TileGrid< CoordT >::to_string() const
{
    std::string res = "";
    tileidx_t idx = 0;
    for ( const auto& pos : positions_ )
        res += std::to_string( idx++ ) + ": " + pos->to_string() + "\n";

    return res;
}


template < typename CoordT >
inline bool TileGrid< CoordT >::operator==( const TileGrid& tg ) const
{
    if ( dimensions_ != tg.dimensions_ )
        return false;

    // if dimensions are equal then
    // total number of tiles and position
    // vector sizes should be equal
    assert( num_tiles_ == tg.num_tiles_ &&
        positions_.size() == tg.positions_.size() );

    return positions_ == tg.positions_;
}
}


#endif
