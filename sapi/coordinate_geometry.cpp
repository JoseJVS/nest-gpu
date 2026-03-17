/*
 *  coordinate_geometry.cpp
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

#include "coordinate_geometry.h"


namespace sapi
{
Coord2D circumscribed_center(
    const Coord2D& coordA,
    const Coord2D& coordB,
    const Coord2D& coordC
)
{
    // circumscribed center ( x, y ) of triangle with vertices A, B, C is such that
    //   (x - xA)^2 + (y - yA)^2
    // = (x - xB)^2 + (y - yB)^2
    // = (x - xC)^2 + (y - yC)^2
    // < = >
    //   x^2 -2xxA + xA^2 + y^2 -2yyA + yA^2
    // = x^2 -2xxB + xB^2 + y^2 -2yyB + yB^2
    // = x^2 -2xxC + xC^2 + y^2 -2yyC + yC^2
    // < = >
    //   -2xxA + xA^2 -2yyA + yA^2
    // = -2xxB + xB^2 -2yyB + yB^2
    // = -2xxC + xC^2 -2yyC + yC^2
    // let D = -2xxA + xA^2 -2yyA + yA^2
    // D = -2xxA + xA^2 -2yyA + yA^2
    // D = -2xxB + xB^2 -2yyB + yB^2
    // D = -2xxC + xC^2 -2yyC + yC^2
    // < = >
    // D + 2xxA + 2yyA = xA^2 + yA^2
    // D + 2xxB + 2yyB = xB^2 + yB^2
    // D + 2xxC + 2yyC = xC^2 + yC^2
    // as a matrix equation system
    // | 1 2xA 2yA |   | D |   | xA^2 + yA^2 |
    // | 1 2xB 2yB | * | x | = | xB^2 + yB^2 |
    // | 1 2xC 2yC |   | y |   | xC^2 + yC^2 |
    // which is solved with Cramer's rule by
    //           | xA^2 + yA^2 2xA 2yA |          | 1 2xA 2yA |
    // D = det ( | xB^2 + yB^2 2xB 2yB | ) / det( | 1 2xB 2yB | )
    //           | xC^2 + yC^2 2xC 2yC |          | 1 2xC 2yC |
    //           | 1 xA^2 + yA^2 2yA |          | 1 2xA 2yA |
    // x = det ( | 1 xB^2 + yB^2 2yB | ) / det( | 1 2xB 2yB | )
    //           | 1 xC^2 + yC^2 2yC |          | 1 2xC 2yC |
    //           | 1 2xA xA^2 + yA^2 |          | 1 2xA 2yA |
    // y = det ( | 1 2xB xB^2 + yB^2 | ) / det( | 1 2xB 2yB | )
    //           | 1 2xC xC^2 + yC^2 |          | 1 2xC 2yC |

    const Coord2D twoA = coordA * 2;
    const Coord2D twoB = coordB * 2;
    const Coord2D twoC = coordC * 2;

    const space_t det_denominator = determinant3x3(
        1, twoA.x_, twoA.y_,
        1, twoB.x_, twoB.y_,
        1, twoC.x_, twoC.y_
    );

    const bool dd_az = almost_zero( det_denominator );

    const space_t n2A = vector_norm2( coordA );
    const space_t n2B = vector_norm2( coordB );
    const space_t n2C = vector_norm2( coordC );

    const space_t det_x_numerator = determinant3x3(
        1, n2A, twoA.y_,
        1, n2B, twoB.y_,
        1, n2C, twoC.y_
    );

    const bool dxn_az = almost_zero( det_x_numerator );

    const space_t det_y_numerator = determinant3x3(
        1, twoA.x_, n2A,
        1, twoB.x_, n2B,
        1, twoC.x_, n2C
    );

    const bool dyn_az = almost_zero( det_y_numerator );

    assert( !dd_az || ( dxn_az && dyn_az ) );

    if ( dd_az || ( dxn_az && dyn_az ) )
        return Coord2D();

    const space_t coord_x = dxn_az
        ? 0
        : almost_equal( det_x_numerator, det_denominator )
        ? 1
        : clamp_epsilon_0( det_x_numerator / det_denominator );

    const space_t coord_y = dyn_az
        ? 0
        : almost_equal( det_y_numerator, det_denominator )
        ? 1
        : clamp_epsilon_0( det_y_numerator / det_denominator );

    return Coord2D( std::move( coord_x ), std::move( coord_y ) );
}


Coord3D circumscribed_center(
    const Coord3D& coordA,
    const Coord3D& coordB,
    const Coord3D& coordC
)
{
    // Here we compute the 2D plane intersecting A, B, and C
    const Coord3D vBA = coordB - coordA;
    const Coord3D vCA = coordC - coordA;

    // A is now the origin of our plane
    // We now compute the basis vectors of the plane
    const Coord3D basis_1 = normalize_vector( vBA );
    assert( !basis_1.is_null() );

    // Gram-Schmidt process for orthonormal basis
    const Coord3D basis_2 = normalize_vector( vCA - projection_vector( vCA, basis_1, false ) );
    assert( !basis_2.is_null() );

    // Redefine A, B, and C as Coord2D in the plane and
    // compute the circumscribed center in 2D
    const Coord2D planar_circumscribed_center = circumscribed_center(
        Coord2D(), // A is the origin of the plane
        Coord2D(
            vector_dot( vBA, basis_1 ),
            vector_dot( vBA, basis_2 )
        ),
        Coord2D(
            vector_dot( vCA, basis_1 ),
            vector_dot( vCA, basis_2 )
        )
    );

    // project the center back to the 3D space
    const Coord3D x_projection = basis_1 * planar_circumscribed_center.x_;
    const Coord3D y_projection = basis_2 * planar_circumscribed_center.y_;
    return Coord3D(
        compensated_sum( coordA.x_, x_projection.x_, y_projection.x_ ),
        compensated_sum( coordA.y_, x_projection.y_, y_projection.y_ ),
        compensated_sum( coordA.z_, x_projection.z_, y_projection.z_ )
    );
}
}
