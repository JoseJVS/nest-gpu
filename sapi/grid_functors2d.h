#ifndef GRID_FUNCTORS2D_H
#define GRID_FUNCTORS2D_H

#include "tiles2d.h"
#include "grid_functors.h"


namespace sapi
{
// Forward definition to avoid linking
Coord2D create_angular_offset( const space_t& );
Coord2D rotate_displacement( const Coord2D&, const Coord2D& );


template < typename SOC2D >
struct ShiftedOriginCreator2D : public BaseShiftedOriginCreator< Coord2D >
{
    const std::optional< Coord2D > angular_offset_;

    ShiftedOriginCreator2D() = delete;
    ShiftedOriginCreator2D( const ShiftedOriginCreator2D& ) = default;
    ShiftedOriginCreator2D( ShiftedOriginCreator2D&& ) = default;

    ShiftedOriginCreator2D(
        const space_t& radius,
        const space_t& rotation
    )
        : BaseShiftedOriginCreator< Coord2D >( radius )
        , angular_offset_(
            almost_zero( rotation )
            ? std::optional< Coord2D >()
            : create_angular_offset( rotation )
        )
    {
    }

    std::unique_ptr< BaseShiftedOriginCreator< Coord2D > >
        clone() const override
    {
        return std::make_unique< SOC2D >( *dynamic_cast< const SOC2D* >( this ) );
    }
};


struct SquareSOC : public ShiftedOriginCreator2D < SquareSOC >
{
    const space_t displacement_;

    SquareSOC() = delete;
    SquareSOC( const SquareSOC& ) = default;
    SquareSOC( SquareSOC&& ) = default;

    SquareSOC(
        const space_t& radius,
        const space_t& rotation
    )
        : ShiftedOriginCreator2D( radius, rotation )
        , displacement_( radius* M_SQRT2 ) // R * sqrt( 2 )
    {
    }

    virtual Coord2D create_shifted_origin(
        const Coord2D& grid_origin,
        const GridPosition< Coord2D >& grid_position,
        const GridPositionParity< Coord2D >& grid_position_parity
    ) const override
    {
        Coord2D disp_vector(
            displacement_ * grid_position[ 0 ],
            displacement_ * grid_position[ 1 ]
        );
        disp_vector.sanitize();
        if ( angular_offset_.has_value() )
            return grid_origin + rotate_displacement( disp_vector, angular_offset_.value() );
        else
            return grid_origin + disp_vector;
    }
};


struct TriangleSOC final : public ShiftedOriginCreator2D< TriangleSOC >
{
    const space_t displacement_;

    TriangleSOC() = delete;
    TriangleSOC( const TriangleSOC& ) = default;
    TriangleSOC( TriangleSOC&& ) = default;

    TriangleSOC(
        const space_t& radius,
        const space_t& rotation
    )
        : ShiftedOriginCreator2D( radius, rotation )
        , displacement_( radius* M_SQRT2 ) // R * sqrt( 2 )
    {
    }

    Coord2D create_shifted_origin(
        const Coord2D& grid_origin,
        const GridPosition< Coord2D >& grid_position,
        const GridPositionParity< Coord2D >& grid_position_parity
    ) const override
    {
        Coord2D disp_vector(
            displacement_ * ( grid_position_parity[ 0 ] ? grid_position[ 0 ] / 2 : ( grid_position[ 0 ] - 1 ) / 2 ),
            displacement_ * grid_position[ 1 ]
        );
        disp_vector.sanitize();
        if ( angular_offset_.has_value() )
            return grid_origin + rotate_displacement( disp_vector, angular_offset_.value() );
        else
            return grid_origin + disp_vector;
    }
};


struct HexagonSOC : public ShiftedOriginCreator2D< HexagonSOC >
{
    const space_t vertical_displacement_;
    const space_t horizontal_displacement_;
    const space_t horizontal_offset_;

    HexagonSOC() = delete;
    HexagonSOC( const HexagonSOC& ) = default;
    HexagonSOC( HexagonSOC&& ) = default;

    HexagonSOC(
        const space_t& radius,
        const space_t& rotation
    )
        : ShiftedOriginCreator2D( radius, rotation )
        , vertical_displacement_( 3 * radius / 2 )
        // With side length a, height h of eq triangle is equal to sqrt( 3 ) * a / 2
        // To reach neighboring hexagon center we need 2 * h
        , horizontal_displacement_( std::sqrt( space_t( 3 ) )* radius )
        , horizontal_offset_( horizontal_displacement_ / 2 )
    {
    }

    Coord2D create_shifted_origin(
        const Coord2D& grid_origin,
        const GridPosition< Coord2D >& grid_position,
        const GridPositionParity< Coord2D >& grid_position_parity
    ) const override
    {
        Coord2D disp_vector(
            grid_position_parity[ 1 ]
            ? horizontal_displacement_ * grid_position[ 0 ]
            : compensated_sum( horizontal_offset_, horizontal_displacement_ * grid_position[ 0 ] ),
            vertical_displacement_ * grid_position[ 1 ]
        );
        disp_vector.sanitize();
        if ( angular_offset_.has_value() )
            return grid_origin + rotate_displacement( disp_vector, angular_offset_.value() );
        else
            return grid_origin + disp_vector;
    }
};


template < typename CTC2D >
struct BaseCachedTile2DCreator : public BaseCachedTileCreator< Coord2D >
{
    const space_t rotation_;

    BaseCachedTile2DCreator() = delete;
    BaseCachedTile2DCreator( const BaseCachedTile2DCreator& ) = default;
    BaseCachedTile2DCreator( BaseCachedTile2DCreator&& ) = default;

    BaseCachedTile2DCreator(
        const space_t& radius,
        const space_t& rotation
    )
        : BaseCachedTileCreator< Coord2D >( radius )
        , rotation_( rotation )
    {
    }

    std::unique_ptr< BaseCachedTileCreator< Coord2D > >
        clone() const override
    {
        return std::make_unique< CTC2D >( *dynamic_cast< const CTC2D* >( this ) );
    }
};


struct SquareCTC : public BaseCachedTile2DCreator< SquareCTC >
{
    SquareCTC() = delete;
    SquareCTC( const SquareCTC& ) = default;
    SquareCTC( SquareCTC&& ) = default;

    SquareCTC(
        const space_t& radius,
        const space_t& rotation
    )
        // For grid alignment base square rotation is 45 degrees
        : BaseCachedTile2DCreator( radius, compensated_sum( rotation, 45. ) )
    {
    }

    bool check_dimensions(
        const GridPosition< Coord2D >& grid_dimensions,
        const bool& edge_wrap
    ) const override
    {
        if ( ( edge_wrap || ( grid_dimensions[ 0 ] + grid_dimensions[ 1 ] > 2 ) )
            && ( int64_t( rotation_ ) % 45 != 0 ) )
            return false;

        return true;
    }

    std::unique_ptr< Tile< Coord2D > > create_tile(
        const Coord2D& tile_origin,
        const GridPositionParity< Coord2D >&
    ) const override
    {
        return std::make_unique< Square >( tile_origin, radius_, rotation_ );
    }
};


struct TriangleCTC : public BaseCachedTile2DCreator< TriangleCTC >
{
    const space_t mirrored_rotation_;

    TriangleCTC() = delete;
    TriangleCTC( const TriangleCTC& ) = default;
    TriangleCTC( TriangleCTC&& ) = default;

    TriangleCTC(
        const space_t& radius,
        const space_t& rotation
    )
        // For grid alignment base triangle rotation is 45 degrees
        : BaseCachedTile2DCreator( radius, compensated_sum( rotation, 45. ) )
        // Triangles need to be mirrored every odd position on grid
        , mirrored_rotation_( compensated_sum( rotation_, 180. ) )
    {
    }

    bool check_dimensions(
        const GridPosition< Coord2D >& grid_dimensions,
        const bool& edge_wrap
    ) const override
    {
        const bool many_tiles = grid_dimensions[ 0 ] + grid_dimensions[ 1 ] > 2;
        if ( !edge_wrap && !many_tiles )
            return true;
        if ( ( edge_wrap || many_tiles ) && ( int64_t( mirrored_rotation_ ) % 180 != 0 ) )
            return false;
        if ( edge_wrap && many_tiles && ( grid_dimensions[ 0 ] % 2 != 0 ) )
            return false;

        return true;
    }

    std::unique_ptr< Tile< Coord2D > > create_tile(
        const Coord2D& tile_origin,
        const GridPositionParity< Coord2D >& grid_position_status
    ) const override
    {
        return std::make_unique< Triangle >( tile_origin, radius_,
            grid_position_status[ 0 ] ? rotation_ : mirrored_rotation_ );
    }
};


struct HexagonCTC : public BaseCachedTile2DCreator< HexagonCTC >
{
    HexagonCTC() = delete;
    HexagonCTC( const HexagonCTC& ) = default;
    HexagonCTC( HexagonCTC&& ) = default;

    HexagonCTC(
        const space_t& radius,
        const space_t& rotation
    )
        // For grid alignment base hexagon rotation is 90 degrees
        : BaseCachedTile2DCreator( radius, compensated_sum( rotation, 90. ) )
    {
    }

    bool check_dimensions(
        const GridPosition< Coord2D >& grid_dimensions,
        const bool& edge_wrap
    ) const override
    {
        const bool many_tiles = grid_dimensions[ 0 ] + grid_dimensions[ 1 ] > 2;
        if ( !edge_wrap && !many_tiles )
            return true;
        if ( ( edge_wrap || many_tiles ) && ( int64_t( rotation_ ) % 30 != 0 ) )
            return false;
        if ( edge_wrap && many_tiles && ( grid_dimensions[ 1 ] % 2 != 0 ) )
            return false;

        return true;
    }

    std::unique_ptr< Tile< Coord2D > > create_tile(
        const Coord2D& tile_origin,
        const GridPositionParity< Coord2D >&
    ) const override
    {
        return std::make_unique< Hexagon >( tile_origin, radius_, rotation_ );
    }
};
}


#endif
