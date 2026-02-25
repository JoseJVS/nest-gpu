#ifndef GRID_CONTAINERS_H
#define GRID_CONTAINERS_H

#include <array>
#include <vector>
#include <cassert>

#include "sapi_config.h"


namespace sapi
{
// Relative displacement in a grid in the form of
// an integer (signed type shift_t) along each dimension of the grid
template < typename CoordT >
using GridShift = std::array< shift_t, CoordT::D >;

// Forward list of relative displacements
template < typename CoordT >
using GridShiftVector = std::vector< GridShift< CoordT > >;

// Position dependent pair GridShiftVector
// first is used of even grid positions
// second is used for odd grid positions
template < typename CoordT >
using GridShiftVectorPair = std::pair< GridShiftVector< CoordT >, GridShiftVector< CoordT > >;

// Array of position dependent GridShiftFLs
// one GridShiftVectorPair for each dimension of the grid
template < typename CoordT >
using GridDimensionalShifts = std::array< GridShiftVectorPair< CoordT >, CoordT::D >;

// Dimensional index decomposition of position in tile grid
template < typename CoordT >
using GridPosition = std::array< tileidx_t, CoordT::D >;

// For each dimensional index
// indicates whether the position is even or odd
template < typename CoordT >
using GridPositionParity = std::array< bool, CoordT::D >;


constexpr bool positiveTix( const tileidx_t& v )
{
    return 0 < v;
}


constexpr bool evenTix( const tileidx_t& v )
{
    return !( v % 2 );
}


template < typename T >
constexpr T position_to_index(
    const std::array< T, 2 >& grid_position,
    const std::array< T, 2 >& grid_dimensions
)
{
    return grid_position[ 0 ] + grid_dimensions[ 0 ] * grid_position[ 1 ];
}


template < typename T >
constexpr T position_to_index(
    const std::array< T, 3 >& grid_position,
    const std::array< T, 3 >& grid_dimensions
)
{
    return grid_position[ 0 ] +
        grid_dimensions[ 0 ] * ( grid_position[ 1 ] +
            grid_dimensions[ 1 ] * grid_position[ 2 ] );
}


constexpr std::pair< tileidx_t, bool > edge_wrapped_shift(
    const tileidx_t& origin,
    const tileidx_t& shift,
    const tileidx_t& upper_edge
)
{
    assert( 0 <= origin && origin < upper_edge );

    const auto shifted_origin = origin + shift;
    if ( shift < 0 )
    {
        if ( shifted_origin < 0 )
        {
            const auto so_mod_u = shifted_origin % upper_edge;
            return { upper_edge * ( so_mod_u != 0 ) + so_mod_u, true };
        }
        else
            return { shifted_origin, false };
    }
    else
    {
        if ( upper_edge <= shifted_origin )
            return { shifted_origin % upper_edge, true };
        else return { shifted_origin, false };
    }
}
}


#endif
