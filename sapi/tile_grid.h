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
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include "tile.h"
#include "bounding_box.h"
#include "grid_containers.h"


namespace sapi
{
template < typename CoordT >
struct ShiftedImage
{
    GridPosition< CoordT > shifted_position_;
    std::unique_ptr< Tile< CoordT > > shifted_tile_;
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
inline ShiftedImage< CoordT >& ShiftedImage< CoordT >::operator=( ShiftedImage&& si )
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
        bool( shifted_tile_ ) == bool( si.shifted_tile_ ) &&
        ( ( bool( shifted_tile_ ) && *shifted_tile_ == *si.shifted_tile_ ) || !bool( shifted_tile_ ) );
}


template < typename CoordT >
struct TilePosition
{
    GridPosition< CoordT > position_;
    std::unique_ptr< Tile< CoordT > > tile_;
    std::unordered_set< tileidx_t > tile_neighborhood_;
    std::vector< ShiftedImage< CoordT > > grid_images_;

    TilePosition() = default;
    TilePosition( const TilePosition& ) = delete;
    TilePosition( TilePosition&& ) = default;
    ~TilePosition() = default;

    TilePosition& operator=( TilePosition&& );

    const Tile< CoordT >* get_tile() const;

    std::string to_string() const;

    bool operator==( const TilePosition& ) const;
};


template < typename CoordT >
inline TilePosition< CoordT >& TilePosition< CoordT >::operator=( TilePosition&& tp )
{
    position_ = std::move( tp.position_ );
    tile_ = std::move( tp.tile_ );
    tile_neighborhood_ = std::move( tp.tile_neighborhood_ );
    grid_images_ = std::move( tp.grid_images_ );

    return *this;
}


template < typename CoordT >
inline const Tile< CoordT >* TilePosition< CoordT >::get_tile() const
{
    assert( tile_ );
    return tile_.get();
}


template < typename CoordT >
std::string TilePosition< CoordT >::to_string() const
{
    assert( tile_ );
    std::string res = tile_->c_radius_.origin_.to_string() + ": [ ";
    auto num_neighbors = tile_neighborhood_.size();
    for ( const auto& neighbor_index : tile_neighborhood_ )
        res += std::to_string( neighbor_index ) +
        ( num_neighbors-- > 1 ? ", " : " ]" );

    return res;
}


template < typename CoordT >
inline bool TilePosition< CoordT >::operator==( const TilePosition& tp ) const
{
    return position_ == tp.position_ &&
        *tile_ == *tp.tile_ &&
        tile_neighborhood_ == tp.tile_neighborhood_ &&
        grid_images_ == tp.grid_images_;
}


template < typename CoordT >
struct TileGrid
{
    split_t splits_ = 0;
    bool has_split_ = false;
    tileidx_t num_tiles_ = 0;
    bool is_edge_wrapped_ = false;
    GridPosition< CoordT > dimensions_;
    BoundingBox< CoordT > bounding_box_;
    std::vector< TilePosition< CoordT > > positions_;

    TileGrid() = default;
    TileGrid( const TileGrid& ) = delete;
    TileGrid( TileGrid&& ) = default;
    ~TileGrid() = default;

    TileGrid( const GridPosition < CoordT >& dimensions );

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
    num_tiles_ = std::accumulate(
        dimensions_.cbegin(),
        dimensions_.cend(),
        num_tiles_,
        std::multiplies< tileidx_t >()
    );
    if ( num_tiles_ < 1 )
        throw std::invalid_argument( "Invalid grid dimensions" );
    positions_.resize( num_tiles_ );
}


template < typename CoordT >
TileGrid< CoordT >& TileGrid< CoordT >::operator=( TileGrid&& tg )
{
    splits_ = tg.splits_;
    tg.splits_ = 0;
    has_split_ = tg.has_split_;
    tg.has_split_ = false;
    num_tiles_ = tg.num_tiles_;
    tg.num_tiles_ = 0;
    is_edge_wrapped_ = tg.is_edge_wrapped_;
    tg.is_edge_wrapped_ = false;

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
    if ( has_split_ )
        throw std::runtime_error( "Tile grid already split" );
    if ( num_splits < 0 )
        throw std::invalid_argument( "Cannot negatively split" );
    if ( owned_tiles.empty() )
        throw std::invalid_argument( "Empty owned tiles" );
    if ( positions_.empty() )
        throw std::invalid_argument( "Cannot split an empty grid" );

#pragma omp parallel default( none )\
shared( positions_, owned_tiles, num_splits )
#pragma omp master
#pragma omp taskgroup
    for ( const auto& position : owned_tiles )
        positions_.at( position ).tile_->initialize_sub_tiles( num_splits );

    splits_ = num_splits;
    has_split_ = true;
}


template < typename CoordT >
std::string TileGrid< CoordT >::to_string() const
{
    std::string res = "";
    tileidx_t idx = 0;
    for ( const auto& pos : positions_ )
    {
        assert( pos );
        res += std::to_string( idx++ ) + ": " + pos->to_string() + "\n";
    }

    return res;
}


template < typename CoordT >
bool TileGrid< CoordT >::operator==( const TileGrid& tg ) const
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
