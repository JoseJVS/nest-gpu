#ifndef TYPE_ERASURE_HELPERS_H
#define TYPE_ERASURE_HELPERS_H

#include <memory>
#include <vector>
#include <stdexcept>

#include "sapi_config.h"


namespace sapi
{
template < typename T >
struct Clonable
{
    virtual ~Clonable() = default;
    virtual std::unique_ptr< T > clone() const = 0;
};


template < typename RT >
struct StateLessCreator
{
    virtual ~StateLessCreator() = default;

    virtual std::unique_ptr< RT > create() const
    {
        throw std::runtime_error( "Undefined creation method for state-less creator" );
    }

    virtual std::unique_ptr< RT > create( const std::vector< space_t >& params ) const
    {
        throw std::runtime_error( "Undefined creation method for state-less creator" );
    }
};
}


#endif
