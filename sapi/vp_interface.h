#ifndef VP_INTERFACE_HPP
#define VP_INTERFACE_HPP

#include <vector>
#include <stdexcept>
#include <cassert>

#include "sapi_config.h"

#ifdef HAVE_MPI
#include <mpi.h>
#endif

#ifdef HAVE_OMP
#include <omp.h>
#endif


namespace sapi
{
inline void init_mpi( vp_t* const& argc, char*** const& argv )
{
#ifdef HAVE_MPI
    vp_t initialized;
    MPI_Initialized( &initialized );
    if ( !initialized )
    {
        vp_t provided_thread_level;
        MPI_Init_thread( argc, argv, MPI_THREAD_FUNNELED, &provided_thread_level );
    }
#endif
}


inline void finalize_mpi()
{
#ifdef HAVE_MPI
    vp_t finalized;
    MPI_Finalized( &finalized );
    if ( !finalized )
        MPI_Finalize();
#endif
}


inline void mpi_barrier()
{
#ifdef HAVE_MPI
    MPI_Barrier( MPI_COMM_WORLD );
#endif
}


inline vp_t get_mpi_rank()
{
#ifdef HAVE_MPI
    vp_t rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    return static_cast< vp_t >( rank );
#else
    return 0;
#endif
}


inline vp_t get_num_mpi_processes()
{
#ifdef HAVE_MPI
    vp_t size;
    MPI_Comm_size( MPI_COMM_WORLD, &size );
    return static_cast< vp_t >( size );
#else
    return 1;
#endif
}


inline void init_omp( vp_t num_threads = 0 )
{
#ifdef HAVE_OMP
    omp_set_dynamic( false );
    if ( num_threads == 0 )
        num_threads = omp_get_max_threads();
    omp_set_num_threads( num_threads );
#endif
}


inline vp_t get_max_omp_threads()
{
#ifdef HAVE_OMP
    return static_cast< vp_t >( omp_get_max_threads() );
#else
    return 1;
#endif
}


inline void set_max_omp_threads( const vp_t& num_threads )
{
#ifdef HAVE_OMP
    if ( num_threads < 1 || 1 << 31 < num_threads )
        throw std::invalid_argument( "Incorrect number of threads to set" );
    omp_set_num_threads( num_threads );
#endif
}


inline vp_t get_thread_num()
{
#ifdef HAVE_OMP
    return static_cast< vp_t >( omp_get_thread_num() );
#else
    return 0;
#endif
}
}


#endif
