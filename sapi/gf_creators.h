/*
 *  gf_creators.h
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

#ifndef GF_CREATORS_H
#define GF_CREATORS_H

#include "grid_functors2d.h"
#include "creator_registry.h"


namespace sapi
{
struct SquareGSCreator : public StateLessCreator< GridTargetPositionShifts< Coord2D > >
{
    std::unique_ptr< GridTargetPositionShifts< Coord2D > > create() const override
    {
        return std::make_unique< GridTargetPositionShifts< Coord2D > >(
            GridShiftVector< Coord2D >{
                { -1, 1 }, { 0, 1 }, { 1, 1 },
                { -1, 0 }, { 0, 0 }, { 1, 0 },
                { -1, -1 }, { 0, -1 }, { 1, -1 }
            },
            GridDimensionalShifts< Coord2D >{}
        );
    }
};


struct TriangleGSCreator : public StateLessCreator< GridTargetPositionShifts< Coord2D > >
{
    std::unique_ptr< GridTargetPositionShifts< Coord2D > > create() const override
    {
        return std::make_unique< GridTargetPositionShifts< Coord2D > >(
            GridShiftVector< Coord2D >{
                // down, up, left, right
                { 0, -1 }, { 0, 1 }, { -1, 0 }, { 1, 0 },
                // 2 * left, 2 * right, up + right, down + left
                { -2, 0 }, { 2, 0 }, { 1, 1 }, { -1, -1 },
                // up + 2 * right, down + 2 * left, self
                { 2, 1 }, { -2, -1 }, { 0, 0 }
            },
            GridDimensionalShifts< Coord2D >{
                GridShiftVectorPair< Coord2D >{
                    GridShiftVector< Coord2D >{
                        // if x pos is even, up + left, up +  3 * right
                        { -1, 1 }, { 3, 1 },
                    },
                    GridShiftVector< Coord2D >{
                        // if x pos is odd, down + right, down + 3 * left
                        { 1, -1 }, { -3, -1 }
                    }
                },
                GridShiftVectorPair< Coord2D >{}
            }
        );
    }
};


struct HexagonGSCreator : public StateLessCreator< GridTargetPositionShifts< Coord2D > >
{
    std::unique_ptr< GridTargetPositionShifts< Coord2D > > create() const override
    {
        return std::make_unique< GridTargetPositionShifts< Coord2D > >(
            GridShiftVector< Coord2D >{
            // down, up, left, right, self
                { 0, -1 }, { 0, 1 }, { -1, 0 }, { 1, 0 }, { 0, 0 }
            },
            GridDimensionalShifts< Coord2D >{
                GridShiftVectorPair< Coord2D >{},
                GridShiftVectorPair< Coord2D >{
                    GridShiftVector< Coord2D >{
                        // if y pos is even, up + left, down + left
                        { -1, 1 }, { -1, -1 }
                    },
                    GridShiftVector< Coord2D >{
                        // if y pos is odd, up + right, down + right
                        { 1, 1 }, { 1, -1 }
                    }
                }
            }
        );
    }
};


template < typename Base, typename Spec >
struct BaseGF2DCreator : public StateLessCreator< Base >
{
    std::unique_ptr< Base > create(
        const std::vector< space_t >& params
    ) const override
    {
        if ( params.size() != 2 || almost_zero( params[ 0 ] ) )
            throw std::invalid_argument( "Incorrect grid functor 2D params" );
        return std::make_unique< Spec >( params[ 0 ], params[ 1 ] );
    }
};


inline void initialize_gsc_registry( CreatorRegistry< GridTargetPositionShifts< Coord2D > >& gscr )
{
    gscr.register_creator< SquareGSCreator >( "Square" );
    gscr.register_creator< TriangleGSCreator >( "Triangle" );
    gscr.register_creator< HexagonGSCreator >( "Hexagon" );
}


inline void initialize_gsc_registry( CreatorRegistry< GridTargetPositionShifts< Coord3D > >& gscr )
{
    throw std::runtime_error( "3D GridGenerator not yet implemented" );
}


inline void initialize_soc_registry( CreatorRegistry< BaseShiftedOriginCreator< Coord2D > >& socr )
{
    socr.register_creator< BaseGF2DCreator< BaseShiftedOriginCreator< Coord2D >, SquareSOC > >( "Square" );
    socr.register_creator< BaseGF2DCreator< BaseShiftedOriginCreator< Coord2D >, TriangleSOC > >( "Triangle" );
    socr.register_creator< BaseGF2DCreator< BaseShiftedOriginCreator< Coord2D >, HexagonSOC > >( "Hexagon" );
}


inline void initialize_soc_registry( CreatorRegistry< BaseShiftedOriginCreator< Coord3D > >& socr )
{
    throw std::runtime_error( "3D GridGenerator not yet implemented" );
}


inline void initialize_ctc_registry( CreatorRegistry< BaseCachedTileCreator< Coord2D > >& ctcr )
{
    ctcr.register_creator< BaseGF2DCreator< BaseCachedTileCreator< Coord2D >, SquareCTC > >( "Square" );
    ctcr.register_creator< BaseGF2DCreator< BaseCachedTileCreator< Coord2D >, TriangleCTC > >( "Triangle" );
    ctcr.register_creator< BaseGF2DCreator< BaseCachedTileCreator< Coord2D >, HexagonCTC > >( "Hexagon" );
}


inline void initialize_ctc_registry( CreatorRegistry< BaseCachedTileCreator< Coord3D > >& ctcr )
{
    throw std::runtime_error( "3D GridGenerator not yet implemented" );
}
}


#endif
