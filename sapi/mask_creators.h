#ifndef MASK_CREATORS_H
#define MASK_CREATORS_H

#include "masks2d.h"
#include "creator_registry.h"


namespace sapi
{
struct CircularMaskCreator : public StateLessCreator< Mask< Coord2D > >
{
    std::unique_ptr< Mask< Coord2D > > create(
        const std::vector< space_t >& params,
        const std::vector< space_t >& offset
    ) const override
    {
        std::unique_ptr< Mask< Coord2D > > mask_ptr;
        switch ( params.size() )
        {
        case 1:
        {
            if ( almost_zero( squared( params[ 0 ] ) ) ||
                std::signbit( params[ 0 ] ) )
                throw std::invalid_argument( "Incorrect CircularMask params" );
            mask_ptr = std::make_unique< CircularMask >( params[ 0 ] );
            break;
        }

        case 3:
        {
            if ( almost_zero( squared( params[ 2 ] ) ) ||
                std::signbit( params[ 2 ] ) )
                throw std::invalid_argument( "Incorrect CircularMask params" );
            mask_ptr = std::make_unique< CircularMask >(
                Coord2D( params[ 0 ], params[ 1 ] ),
                params[ 2 ]
            );
            break;
        }

        default:
            throw std::invalid_argument( "Incorrect CircularMask params" );
            break;
        }

        if ( !offset.empty() )
        {
            if ( offset.size() != static_cast< std::size_t >( Coord2D::D ) )
                throw std::invalid_argument( "Incorrect offset params" );
            mask_ptr->set_offset( Coord2D::copy_from_vec( offset.cbegin() ) );
        }

        return mask_ptr;
    }
};


struct ParallelogramMaskCreator : public StateLessCreator< Mask< Coord2D > >
{
    std::unique_ptr< Mask< Coord2D > > create(
        const std::vector< space_t >& params,
        const std::vector< space_t >& offset
    ) const override
    {
        std::unique_ptr< Mask< Coord2D > > mask_ptr;
        switch ( params.size() )
        {
        case 4:
        {
            Coord2D axis0( params[ 0 ], params[ 1 ] );
            Coord2D axis1( params[ 2 ], params[ 3 ] );
            if (
                almost_zero( vector_norm2( axis0 ) ) ||
                almost_zero( vector_norm2( axis1 ) ) ||
                almost_zero( coord_sum( vector_cross( axis0, axis1 ) ) )
                )
                throw std::invalid_argument( "Incorrect ParallelogramMask params" );
            mask_ptr = std::make_unique< ParallelogramMask >(
                std::move( axis0 ),
                std::move( axis1 )
            );
            break;
        }

        case 6:
        {
            Coord2D axis0( params[ 2 ], params[ 3 ] );
            Coord2D axis1( params[ 4 ], params[ 5 ] );
            if (
                almost_zero( vector_norm2( axis0 ) ) ||
                almost_zero( vector_norm2( axis1 ) ) ||
                almost_zero( coord_sum( vector_cross( axis0, axis1 ) ) )
                )
                throw std::invalid_argument( "Incorrect ParallelogramMask params" );
            mask_ptr = std::make_unique< ParallelogramMask >(
                Coord2D( params[ 0 ], params[ 1 ] ),
                std::move( axis0 ),
                std::move( axis1 )
            );
            break;
        }

        default:
            throw std::invalid_argument( "Incorrect ParallelogramMask params" );
            break;
        }

        if ( !offset.empty() )
        {
            if ( offset.size() != static_cast< std::size_t >( Coord2D::D ) )
                throw std::invalid_argument( "Incorrect offset params" );
            mask_ptr->set_offset( Coord2D::copy_from_vec( offset.cbegin() ) );
        }

        return mask_ptr;
    }
};


struct EllipticalMaskCreator : public StateLessCreator< Mask< Coord2D > >
{
    std::unique_ptr< Mask< Coord2D > > create(
        const std::vector< space_t >& params,
        const std::vector< space_t >& offset
    ) const override
    {
        std::unique_ptr< Mask< Coord2D > > mask_ptr;
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
            mask_ptr = std::make_unique< EllipticalMask >(
                params[ 0 ], params[ 1 ]
            );
            break;
        }

        case 3:
        {
            if ( almost_zero( squared( params[ 0 ] ) ) ||
                almost_zero( squared( params[ 1 ] ) ) ||
                std::signbit( params[ 0 ] ) ||
                std::signbit( params[ 1 ] ) ||
                !leq_test( params[ 1 ], params[ 0 ] ) )
                throw std::invalid_argument( "Incorrect EllipticalMask params" );
            mask_ptr = std::make_unique< EllipticalMask >(
                params[ 0 ], params[ 1 ], params[ 2 ]
            );
            break;
        }

        case 4:
        {
            if ( almost_zero( squared( params[ 2 ] ) ) ||
                almost_zero( squared( params[ 3 ] ) ) ||
                std::signbit( params[ 2 ] ) ||
                std::signbit( params[ 3 ] ) ||
                !leq_test( params[ 3 ], params[ 2 ] ) )
                throw std::invalid_argument( "Incorrect EllipticalMask params" );
            mask_ptr = std::make_unique< EllipticalMask >(
                Coord2D( params[ 0 ], params[ 1 ] ),
                params[ 2 ], params[ 3 ]
            );
            break;
        }

        case 5:
        {
            if ( almost_zero( squared( params[ 2 ] ) ) ||
                almost_zero( squared( params[ 3 ] ) ) ||
                std::signbit( params[ 2 ] ) ||
                std::signbit( params[ 3 ] ) ||
                !leq_test( params[ 3 ], params[ 2 ] ) )
                throw std::invalid_argument( "Incorrect EllipticalMask params" );
            mask_ptr = std::make_unique< EllipticalMask >(
                Coord2D( params[ 0 ], params[ 1 ] ),
                params[ 2 ], params[ 3 ], params[ 4 ]
            );
            break;
        }

        default:
            throw std::invalid_argument( "Incorrect EllipticalMask params" );
            break;
        }

        if ( !offset.empty() )
        {
            if ( offset.size() != static_cast< std::size_t >( Coord2D::D ) )
                throw std::invalid_argument( "Incorrect offset params" );
            mask_ptr->set_offset( Coord2D::copy_from_vec( offset.cbegin() ) );
        }

        return mask_ptr;
    }
};


struct TriangularMaskCreator : public StateLessCreator< Mask< Coord2D > >
{
    std::unique_ptr< Mask< Coord2D > > create(
        const std::vector< space_t >& params,
        const std::vector< space_t >& offset
    ) const override
    {
        std::unique_ptr< Mask< Coord2D > > mask_ptr;
        switch ( params.size() )
        {
        case 4:
        {
            Coord2D axis0( params[ 0 ], params[ 1 ] );
            Coord2D axis1( params[ 2 ], params[ 3 ] );
            if (
                almost_zero( vector_norm2( axis0 ) ) ||
                almost_zero( vector_norm2( axis1 ) ) ||
                almost_zero( coord_sum( vector_cross( axis0, axis1 ) ) )
                )
                throw std::invalid_argument( "Incorrect TriangularMask params" );
            mask_ptr = std::make_unique< TriangularMask >(
                std::move( axis0 ),
                std::move( axis1 )
            );
            break;
        }

        case 6:
        {
            Coord2D axis0( params[ 2 ], params[ 3 ] );
            Coord2D axis1( params[ 4 ], params[ 5 ] );
            if (
                almost_zero( vector_norm2( axis0 ) ) ||
                almost_zero( vector_norm2( axis1 ) ) ||
                almost_zero( coord_sum( vector_cross( axis0, axis1 ) ) )
                )
                throw std::invalid_argument( "Incorrect TriangularMask params" );
            mask_ptr = std::make_unique< TriangularMask >(
                Coord2D( params[ 0 ], params[ 1 ] ),
                std::move( axis0 ),
                std::move( axis1 )
            );
            break;
        }

        default:
            throw std::invalid_argument( "Incorrect ParallelogramMask params" );
            break;
        }

        if ( !offset.empty() )
        {
            if ( offset.size() != static_cast< std::size_t >( Coord2D::D ) )
                throw std::invalid_argument( "Incorrect offset params" );
            mask_ptr->set_offset( Coord2D::copy_from_vec( offset.begin() ) );
        }

        return mask_ptr;
    }
};


inline void initialize_mk_registry( CreatorRegistry< Mask< Coord2D > >& mkr )
{
    mkr.register_creator< CircularMaskCreator >( "Circular" );
    mkr.register_creator< ParallelogramMaskCreator >( "Parallelogram" );
    mkr.register_creator< EllipticalMaskCreator >( "Elliptical" );
    mkr.register_creator< TriangularMaskCreator >( "Triangular" );
}


inline void initialize_mk_registry( CreatorRegistry< Mask< Coord3D > >& mkr )
{
    throw std::runtime_error( "3D Masks not yet implemented" );
}
}


#endif
