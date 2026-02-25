#ifndef GRID_FUNCTORS_H
#define GRID_FUNCTORS_H

#include "tile.h"
#include "grid_containers.h"
#include "type_erasure_helpers.h"


namespace sapi
{
template < typename CoordT >
struct GridTargetPositionShifts : public Clonable< GridTargetPositionShifts< CoordT > >
{
    // Position independent GridShiftVector and dimension position dependent GridShiftVector
    // for complete displacement definition of position relative grid targets
    const GridShiftVector< CoordT > position_independent_shifts_;
    const GridDimensionalShifts< CoordT > position_dependent_shifts_;

    GridTargetPositionShifts() = delete;
    GridTargetPositionShifts( const GridTargetPositionShifts& ) = delete;
    GridTargetPositionShifts( GridTargetPositionShifts&& ) = default;

    GridTargetPositionShifts(
        const GridShiftVector< CoordT >& pis,
        const GridDimensionalShifts< CoordT >& pds
    )
        : position_independent_shifts_( pis )
        , position_dependent_shifts_( pds )
    {
    }

    GridTargetPositionShifts(
        GridShiftVector< CoordT >&& pis,
        GridDimensionalShifts< CoordT >&& pds
    )
        : position_independent_shifts_( std::move( pis ) )
        , position_dependent_shifts_( std::move( pds ) )
    {
    }

    std::unique_ptr< GridTargetPositionShifts >
        clone() const
    {
        return std::make_unique< GridTargetPositionShifts >(
            position_independent_shifts_,
            position_dependent_shifts_
        );
    }
};


template < typename CoordT >
struct BaseShiftedOriginCreator : public Clonable< BaseShiftedOriginCreator< CoordT > >
{
    BaseShiftedOriginCreator() = delete;
    BaseShiftedOriginCreator( const BaseShiftedOriginCreator& ) = default;
    BaseShiftedOriginCreator( BaseShiftedOriginCreator&& ) = default;

    BaseShiftedOriginCreator( const space_t& radius )
    {
        assert( !almost_zero( radius ) );
    }

    virtual CoordT create_shifted_origin(
        const CoordT& grid_origin,
        const GridPosition< CoordT >& grid_position,
        const GridPositionParity< CoordT >& grid_position_parity
    ) const = 0;
};


template < typename CoordT >
struct BaseCachedTileCreator : public Clonable< BaseCachedTileCreator< CoordT > >
{
    const space_t radius_;

    BaseCachedTileCreator() = delete;
    BaseCachedTileCreator( const BaseCachedTileCreator& ) = default;
    BaseCachedTileCreator( BaseCachedTileCreator&& ) = default;

    BaseCachedTileCreator( const space_t& radius )
        : radius_( radius )
    {
        assert( !almost_zero( radius ) );
    }

    virtual bool check_dimensions(
        const GridPosition< CoordT >& grid_dimensions,
        const bool& edge_wrap
    ) const = 0;

    virtual std::unique_ptr< Tile< CoordT > > create_tile(
        const CoordT& tile_origin,
        const GridPositionParity< CoordT >& tile_position_parity
    ) const = 0;
};
}


#endif
