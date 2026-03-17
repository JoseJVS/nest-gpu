/*
 *  grid_functors.h
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

#ifndef GRID_FUNCTORS_H
#define GRID_FUNCTORS_H

#include "gf_geometry.h"


namespace sapi
{
template < typename CoordT >
struct GridTargetPositionShifts
{
    // Position independent GridShiftVector and dimension position dependent GridShiftVector
    // for complete displacement definition of position relative grid targets
    GridShiftVector< CoordT > position_independent_shifts_;
    GridDimensionalShifts< CoordT > position_dependent_shifts_;

    GridTargetPositionShifts() = default;
    GridTargetPositionShifts( const GridTargetPositionShifts& ) = default;
    GridTargetPositionShifts( GridTargetPositionShifts&& ) = default;
    ~GridTargetPositionShifts() = default;

    GridTargetPositionShifts(
        const GridShiftVector< CoordT >&,
        const GridDimensionalShifts< CoordT >&
    );

    GridTargetPositionShifts(
        GridShiftVector< CoordT >&&,
        GridDimensionalShifts< CoordT >&&
    );

    GridTargetPositionShifts& operator=( GridTargetPositionShifts&& );
};


template < typename CoordT >
GridTargetPositionShifts< CoordT >::GridTargetPositionShifts(
    const GridShiftVector< CoordT >& pis,
    const GridDimensionalShifts< CoordT >& pds
)
    : position_independent_shifts_( pis )
    , position_dependent_shifts_( pds )
{
}


template < typename CoordT >
GridTargetPositionShifts< CoordT >::GridTargetPositionShifts(
    GridShiftVector< CoordT >&& pis,
    GridDimensionalShifts< CoordT >&& pds
)
    : position_independent_shifts_( std::move( pis ) )
    , position_dependent_shifts_( std::move( pds ) )
{
}


template < typename CoordT >
inline GridTargetPositionShifts< CoordT >&
GridTargetPositionShifts< CoordT >::operator=( GridTargetPositionShifts&& g )
{
    position_independent_shifts_ = std::move( g.position_independent_shifts_ );
    position_dependent_shifts_ = std::move( g.position_dependent_shifts_ );

    return *this;
}


template < typename CoordT >
struct ShiftedOriginCreator
{
    TILE_SHAPE shape_ = TILE_SHAPE::NULL_TS;
    std::vector< CoordT > helper_vectors_;
    std::vector< space_t > helper_scalars_;

    ShiftedOriginCreator() = default;
    ShiftedOriginCreator( const ShiftedOriginCreator& ) = default;
    ShiftedOriginCreator( ShiftedOriginCreator&& ) = default;
    ~ShiftedOriginCreator() = default;

    ShiftedOriginCreator(
        const TILE_SHAPE&,
        const std::vector< space_t >&,
        const std::vector< angle_t >&
    );

    ShiftedOriginCreator& operator=( ShiftedOriginCreator&& );

    CoordT create_shifted_origin(
        const CoordT&,
        const GridPosition< CoordT >&,
        const GridPositionParity< CoordT >&
    ) const;
};


template < typename CoordT >
ShiftedOriginCreator< CoordT >::ShiftedOriginCreator(
    const TILE_SHAPE& shape,
    const std::vector< space_t >& side_lengths,
    const std::vector< angle_t >& angular_offsets
)
    : shape_( shape )
{
    sapi::initialize_soc(
        helper_vectors_,
        helper_scalars_,
        side_lengths,
        angular_offsets,
        shape_
    );
}


template < typename CoordT >
inline ShiftedOriginCreator< CoordT >&
ShiftedOriginCreator< CoordT >::operator=( ShiftedOriginCreator&& soc )
{
    shape_ = soc.shape_;
    helper_vectors_ = std::move( soc.helper_vectors_ );
    helper_scalars_ = std::move( soc.helper_scalars_ );

    soc.shape_ = TILE_SHAPE::NULL_TS;

    return *this;
}


template < typename CoordT >
inline CoordT
ShiftedOriginCreator< CoordT >::create_shifted_origin(
    const CoordT& grid_origin,
    const GridPosition< CoordT >& gp,
    const GridPositionParity< CoordT >& gpp
) const
{
    return sapi::create_shifted_origin(
        *this,
        grid_origin,
        gp,
        gpp
    );
}


template < typename CoordT >
struct CachedTileCreator
{
    TILE_SHAPE shape_ = TILE_SHAPE::NULL_TS;
    std::vector< space_t > side_lengths_;
    std::vector< angle_t > angular_offsets_;

    CachedTileCreator() = default;
    CachedTileCreator( const CachedTileCreator& ) = default;
    CachedTileCreator( CachedTileCreator&& ) = default;
    ~CachedTileCreator() = default;

    CachedTileCreator(
        const TILE_SHAPE&,
        const std::vector< space_t >&,
        const std::vector< angle_t >&
    );

    CachedTileCreator& operator=( CachedTileCreator&& );

    bool check_dimensions(
        const GridPosition< CoordT >&
    ) const;

    Tile< CoordT > create_tile(
        const CoordT&,
        const GridPositionParity< CoordT >&
    ) const;
};


template < typename CoordT >
CachedTileCreator< CoordT >::CachedTileCreator(
    const TILE_SHAPE& shape,
    const std::vector< space_t >& side_lengths,
    const std::vector< angle_t >& angular_offsets
)
    : shape_( shape )
{
    sapi::initialize_ctc(
        side_lengths_,
        angular_offsets_,
        side_lengths,
        angular_offsets,
        shape_
    );
}


template < typename CoordT >
inline CachedTileCreator< CoordT >&
CachedTileCreator< CoordT >::operator=( CachedTileCreator&& ctc )
{
    shape_ = ctc.shape_;
    side_lengths_ = std::move( ctc.side_lengths_ );
    angular_offsets_ = std::move( ctc.angular_offsets_ );

    ctc.shape_ = TILE_SHAPE::NULL_TS;

    return *this;
}


template < typename CoordT >
inline bool
CachedTileCreator< CoordT >::check_dimensions(
    const GridPosition< CoordT >& grid_dimensions
) const
{
    return sapi::check_dimensions( *this, grid_dimensions );
}


template < typename CoordT >
inline Tile< CoordT >
CachedTileCreator< CoordT >::create_tile(
    const CoordT& tile_origin,
    const GridPositionParity< CoordT >& gpp
) const
{
    return sapi::create_tile( *this, tile_origin, gpp );
}
}


#endif
