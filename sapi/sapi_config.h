#ifndef CONFIG_H
#define CONFIG_H

#include <cstdint>

#define HAVE_OMP
#define HAVE_MPI


namespace sapi
{
typedef int32_t vp_t; // Virtual process id type
typedef int32_t nodeidx_t; // Needs to be signed to handle overflow
typedef int64_t largenodeidx_t; // Needs to be at least equal or larger than nodeidx_t
typedef int32_t tileidx_t; // Needs to be signed to handle overflow
typedef uint8_t vertidx_t; // Current implemented tiles count with up to 12 vertices
typedef uint8_t split_t; // Needs to be unsigned and exponentially smaller than tileidx_t
typedef uint8_t dim_t; // For static coordinate dimension count
typedef int8_t shift_t; // Needs to be signed and smaller than tileidx_t
typedef int32_t count_t; // Needs to be signed
typedef uint16_t mult_t; // For connection multiplicity
typedef double space_t; // For spatial data
typedef float conn_t; // For connection parameters

constexpr static const uint8_t BATCHING_THRESHOLD = 100;
constexpr static const uint8_t TOLERANCE = 2;
constexpr static const uint8_t RELATIVE_TOLERANCE = 8;
}


#endif
