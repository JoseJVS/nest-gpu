#ifndef MASK_CREATORS_H
#define MASK_CREATORS_H

#include "masks2d.h"
#include "creator_registry.h"


namespace sapi
{
struct CircularMaskCreator : public StateLessCreator< Mask< Coord2D > >
{
    std::unique_ptr< Mask< Coord2D > > create(
        const std::vector< space_t >& params
    ) const override
    {
        switch ( params.size() )
        {
        case 1:
        {
            if ( almost_zero( squared( params[ 0 ] ) ) ||
                std::signbit( params[ 0 ] ) )
                throw std::invalid_argument( "Incorrect CircularMask params" );
            return std::make_unique< CircularMask >( params[ 0 ] );
            break;
        }

        case 3:
        {
            if ( almost_zero( squared( params[ 2 ] ) ) ||
                std::signbit( params[ 2 ] ) )
                throw std::invalid_argument( "Incorrect CircularMask params" );
            return std::make_unique< CircularMask >(
                Coord2D( params[ 0 ], params[ 1 ] ),
                params[ 2 ]
            );
        }

        default:
            throw std::invalid_argument( "Incorrect CircularMask params" );
            break;
        }
    }
};


struct ParallelogramMaskCreator : public StateLessCreator< Mask< Coord2D > >
{
    std::unique_ptr< Mask< Coord2D > > create(
        const std::vector< space_t >& params
    ) const override
    {
        switch ( params.size() )
        {
        case 4:
        {
            Coord2D axis0( params[ 0 ], params[ 1 ] );
            Coord2D axis1( params[ 2 ], params[ 3 ] );
            if (
                almost_zero( vector_norm2( axis0 ) ) ||
                almost_zero( vector_norm2( axis1 ) )
                )
                throw std::invalid_argument( "Incorrect ParallelogramMask params" );
            return std::make_unique< ParallelogramMask >(
                std::move( axis0 ),
                std::move( axis1 )
            );
        }

        case 6:
        {
            Coord2D axis0( params[ 2 ], params[ 3 ] );
            Coord2D axis1( params[ 4 ], params[ 5 ] );
            if (
                almost_zero( vector_norm2( axis0 ) ) ||
                almost_zero( vector_norm2( axis1 ) )
                )
                throw std::invalid_argument( "Incorrect ParallelogramMask params" );
            return std::make_unique< ParallelogramMask >(
                Coord2D( params[ 0 ], params[ 1 ] ),
                std::move( axis0 ),
                std::move( axis1 )
            );
        }

        default:
            throw std::invalid_argument( "Incorrect ParallelogramMask params" );
            break;
        }
    }
};


struct EllipticalMaskCreator : public StateLessCreator< Mask< Coord2D > >
{
    std::unique_ptr< Mask< Coord2D > > create(
        const std::vector< space_t >& params
    ) const override
    {
        switch ( params.size() )
        {
        case 2:
        {
            if ( almost_zero( squared( params[ 0 ] ) ) ||
                almost_zero( squared( params[ 1 ] ) ) ||
                std::signbit( params[ 0 ] ) ||
                std::signbit( params[ 1 ] ) ||
                !leq_test( params[ 1 ], params[ 0 ] ) )
                throw std::invalid_argument( "Incorrect EllipticalMask params" );
            return std::make_unique< EllipticalMask >(
                params[ 0 ], params[ 1 ]
            );
        }

        case 3:
        {
            if ( almost_zero( squared( params[ 0 ] ) ) ||
                almost_zero( squared( params[ 1 ] ) ) ||
                std::signbit( params[ 0 ] ) ||
                std::signbit( params[ 1 ] ) ||
                !leq_test( params[ 1 ], params[ 0 ] ) )
                throw std::invalid_argument( "Incorrect EllipticalMask params" );
            return std::make_unique< EllipticalMask >(
                params[ 0 ], params[ 1 ], params[ 2 ]
            );
        }

        case 4:
        {
            if ( almost_zero( squared( params[ 2 ] ) ) ||
                almost_zero( squared( params[ 3 ] ) ) ||
                std::signbit( params[ 2 ] ) ||
                std::signbit( params[ 3 ] ) ||
                !leq_test( params[ 3 ], params[ 2 ] ) )
                throw std::invalid_argument( "Incorrect EllipticalMask params" );
            return std::make_unique< EllipticalMask >(
                Coord2D( params[ 0 ], params[ 1 ] ),
                params[ 2 ], params[ 3 ]
            );
        }

        case 5:
        {
            if ( almost_zero( squared( params[ 2 ] ) ) ||
                almost_zero( squared( params[ 3 ] ) ) ||
                std::signbit( params[ 2 ] ) ||
                std::signbit( params[ 3 ] ) ||
                !leq_test( params[ 3 ], params[ 2 ] ) )
                throw std::invalid_argument( "Incorrect EllipticalMask params" );
            return std::make_unique< EllipticalMask >(
                Coord2D( params[ 0 ], params[ 1 ] ),
                params[ 2 ], params[ 3 ], params[ 4 ]
            );
        }

        default:
            throw std::invalid_argument( "Incorrect EllipticalMask params" );
            break;
        }
    }
};


inline void initialize_mk_registry( CreatorRegistry< Mask< Coord2D > >& mkr )
{
    mkr.register_creator< CircularMaskCreator >( "Circular" );
    mkr.register_creator< ParallelogramMaskCreator >( "Parallelogram" );
    mkr.register_creator< EllipticalMaskCreator >( "Elliptical" );
}


inline void initialize_mk_registry( CreatorRegistry< Mask< Coord3D > >& mkr )
{
    throw std::runtime_error( "3D Masks not yet implemented" );
}
}


#endif
