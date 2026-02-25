#ifndef THREAD_ALIGNED_ARRAY_H
#define THREAD_ALIGNED_ARRAY_H

#include <vector>
#include <memory>
#include <type_traits>
#include <cassert>

#include "sapi_config.h"
#include "type_erasure_helpers.h"


namespace sapi
{
// Forward definition to link with vp_interface
vp_t get_thread_num();
vp_t get_max_omp_threads();


template < typename T,
    typename std::enable_if_t<
    std::is_base_of_v< Clonable< T >, T >, bool
> = true >
class TAArray
{
public:
    TAArray() = default;
    TAArray( const TAArray& ) = delete;
    TAArray( TAArray&& ) = default;
    ~TAArray() = default;

    bool is_initialized() const;

    void clear();
    void prepare();
    void clone( const T& );
    void clone( const std::unique_ptr< T >& );

    const std::unique_ptr< T >& get_thread_item( const vp_t& ) const;
    const std::unique_ptr< T >& get_local_thread_item() const;

protected:
    bool cloned_ = false;
    bool prepared_ = false;
    vp_t num_threads_ = 0;
    std::vector< std::unique_ptr< T > > item_vec_;
};


template < typename T,
    typename std::enable_if_t<
    std::is_base_of_v< Clonable< T >, T >, bool
> b >
inline void TAArray< T, b >::clear()
{
    item_vec_.clear();
}


template < typename T,
    typename std::enable_if_t<
    std::is_base_of_v< Clonable< T >, T >, bool
> b >
inline bool TAArray< T, b >::is_initialized() const
{
    assert( cloned_ <= prepared_ );;
    return cloned_;
}


template < typename T,
    typename std::enable_if_t<
    std::is_base_of_v< Clonable< T >, T >, bool
> b >
inline void TAArray< T, b >::prepare()
{
    if ( !prepared_ || cloned_ || num_threads_ != get_max_omp_threads() )
    {
        item_vec_.clear();
        num_threads_ = get_max_omp_threads();
        item_vec_.resize( num_threads_ );
        prepared_ = true;
        cloned_ = false;
    }
}


template < typename T,
    typename std::enable_if_t<
    std::is_base_of_v< Clonable< T >, T >, bool
> b >
inline void TAArray< T, b >::clone(
    const T& item
)
{
    prepare();
    assert( !cloned_ && prepared_ );

#pragma omp parallel default( none )\
shared( item_vec_, item )
    {
        const auto tid = get_thread_num();
        item_vec_[ tid ] = item.clone();
    }

    cloned_ = true;
}


template < typename T,
    typename std::enable_if_t<
    std::is_base_of_v< Clonable< T >, T >, bool
> b >
inline void TAArray< T, b >::clone(
    const std::unique_ptr< T >& item
)
{
    prepare();
    assert( !cloned_ && prepared_ );

#pragma omp parallel default( none )\
shared( item_vec_, item )
    {
        const auto tid = get_thread_num();
        item_vec_[ tid ] = item->clone();
    }

    cloned_ = true;
}


template < typename T,
    typename std::enable_if_t<
    std::is_base_of_v< Clonable< T >, T >, bool
> b >
inline const std::unique_ptr< T >&
TAArray< T, b >::get_thread_item( const vp_t& tid ) const
{
    assert( 0 <= tid && tid < num_threads_ );
    return item_vec_[ tid ];
}


template < typename T,
    typename std::enable_if_t<
    std::is_base_of_v< Clonable< T >, T >, bool
> b >
inline const std::unique_ptr< T >&
TAArray< T, b >::get_local_thread_item() const
{
    const auto tid = get_thread_num();
    assert( 0 <= tid && tid < num_threads_ );
    return item_vec_[ tid ];
}
}


#endif
