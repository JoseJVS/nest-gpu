/*
 *  tile2d_geometry.h
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

#ifndef TILE2D_GEOMETRY_H
#define TILE2D_GEOMETRY_H

#include "type_erasure_helpers.h"


namespace sapi
{
// Forward definition to link with coordinates.h
struct Coord2D;
template < typename CoordT >
struct CircumscribedRadius;

// Forward definition to link with tile.h
template < typename CoordT >
struct Tile;


void initialize_vertices_2D(
    std::vector< Coord2D >&,
    CircumscribedRadius< Coord2D >&,
    const std::vector< space_t >&,
    const angle_t&,
    const bool&
);


void initialize_hexagon_vertices(
    std::vector< Coord2D >&,
    CircumscribedRadius< Coord2D >&,
    const space_t&,
    const angle_t&
);


void initialize_helpers_2D(
    std::vector< Coord2D >&,
    std::vector< space_t >&,
    const std::vector< Coord2D >&
);


void initialize_hexagon_helpers(
    std::vector< Coord2D >&,
    std::vector< space_t >&,
    const std::vector< Coord2D >&
);


void split_rectangle(
    std::vector< Tile< Coord2D > >&,
    const std::vector< Coord2D >&,
    const CircumscribedRadius< Coord2D >&,
    const tileidx_t&,
    const split_t&
);


void split_triangle(
    std::vector< Tile< Coord2D > >&,
    const std::vector< Coord2D >&,
    const tileidx_t&,
    const split_t&
);


void split_hexagon(
    std::vector< Tile< Coord2D > >&,
    const std::vector< Coord2D >&,
    const CircumscribedRadius< Coord2D >&,
    const tileidx_t&,
    const split_t&
);


bool coord_in_tile_2D(
    const Coord2D&,
    const CircumscribedRadius< Coord2D >&,
    const std::vector< Coord2D >&,
    const std::vector< Coord2D >&,
    const std::vector< space_t >&,
    const bool&
);


bool coord_in_hexagon(
    const Coord2D&,
    const CircumscribedRadius< Coord2D >&,
    const std::vector< Coord2D >&,
    const std::vector< Coord2D >&,
    const std::vector< space_t >&
);


void generate_coords_in_rectangle(
    std::vector< Coord2D >&,
    AnyRNG&,
    const std::vector< Coord2D >&,
    const std::vector< Coord2D >&
);


void generate_coords_in_triangle(
    std::vector< Coord2D >&,
    AnyRNG&,
    const std::vector< Coord2D >&
);


void generate_coords_in_hexagon(
    std::vector< Coord2D >&,
    AnyRNG&,
    const std::vector< Coord2D >&,
    const CircumscribedRadius< Coord2D >&
);


Coord2D project_point_to_2D_perimeter(
    const Coord2D&,
    const std::vector< Coord2D >&
);
}


#endif
