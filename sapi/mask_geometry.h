#ifndef MASK_GEOMETRY_H
#define MASK_GEOMETRY_H

#include "coordinate_geometry.h"


namespace sapi
{
inline std::optional< Displacement< Coord2D > >
coord_in_circular_mask(
    const Coord2D& mask_origin,
    const space_t& circular_radius2,
    const Coord2D& coord
)
{
    Displacement< Coord2D > disp( coord - mask_origin );
    return leq_test( disp.distance2_, circular_radius2 )
        ? std::make_optional( disp )
        : std::optional< Displacement< Coord2D > >();
}


inline std::optional< Displacement< Coord2D > >
coord_in_algebraic_mask(
    const Coord2D& mask_origin,
    const space_t& circular_radius2,
    const Coord2D& basis_vertex,
    const Coord2D& basis_vector0,
    const Coord2D& basis_vector1,
    const space_t& det_01,
    const Coord2D& coord,
    const bool& triangular_comparison
)
{
    // Get displacement from mask origin
    Displacement< Coord2D > disp( coord - mask_origin );

    // Fast rejection method
    if ( !leq_test( disp.distance2_, circular_radius2 ) )
        return {};

    // Get projection coefficients of coord onto parallelogram basis vectors
    return algebraic_projection_comparison(
        coord - basis_vertex,
        basis_vector0,
        basis_vector1,
        det_01,
        triangular_comparison
    )
        ? std::make_optional( std::move( disp ) )
        : std::optional< Displacement< Coord2D > >();
}


inline bool rotated_coord_in_elliptical_mask(
    const Coord2D& displacement,
    const space_t& semi_major_axe2,
    const space_t& semi_minor_axe2
)
{
    const bool dx_az = almost_zero( displacement.x_ );
    const bool dy_az = almost_zero( displacement.y_ );

    if ( dx_az && dy_az )
        return true;

    if ( dx_az )
        return leq_test( squared( displacement.y_ ), semi_minor_axe2 );

    if ( dy_az )
        return leq_test( squared( displacement.x_ ), semi_major_axe2 );

    return leq_test(
        compensated_sum(
            squared( displacement.x_ ) / semi_major_axe2,
            squared( displacement.y_ ) / semi_minor_axe2
        ),
        1
    );
}


inline std::optional< Displacement< Coord2D > >
coord_in_elliptical_mask(
    const Coord2D& mask_origin,
    const space_t& semi_major_axe2,
    const space_t& semi_minor_axe2,
    const std::optional< Coord2D >& angular_offset,
    const Coord2D& coord
)
{
    // Get displacement from mask origin
    Displacement< Coord2D > disp( coord - mask_origin );

    // Fast rejection method
    if ( !leq_test( disp.distance2_, semi_major_axe2 ) )
        return {};

    // Given ellipse with h, k center, a semi major, b semi minor, @ rotation
    // If x, y point is in ellipse then 
    // X^2 / a^2 + Y^2 / b^2 <= 1
    // With X = ( x - h ) * cos @ + ( y - k ) * sin @
    // and  Y = - ( x - h ) * sin @ + ( y - k ) * cos @ 
    return rotated_coord_in_elliptical_mask(
        angular_offset.has_value()
        ? rotate_displacement( disp.displacement_, angular_offset.value() )
        : disp.displacement_,
        semi_major_axe2,
        semi_minor_axe2
    )
        ? std::make_optional( std::move( disp ) )
        : std::optional< Displacement< Coord2D > >();
}
}


#endif
