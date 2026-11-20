/*
 *  algebraic_functors.h
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

#ifndef ALGEBRAIC_FUNCTORS_H
#define ALGEBRAIC_FUNCTORS_H

#include <cassert>

#include "enum_store.h"
#include "numerics.h"
#include "sapi_forward_declarations.h"

namespace sapi {
// Forward definition to link with coordinates.h
struct Coord2D;
struct Coord3D;

struct UnaryFunctor {
  UNARY_FUNCTION func_ = UNARY_FUNCTION::NULL_UF;
  space_t var_[2] = {0.};

  inline space_t operator()(const space_t x) const;
};

inline UnaryFunctor construct_unary_functor(const UNARY_FUNCTION function,
                                            const space_t var0 = 0.,
                                            const space_t var1 = 0.) {
  return {.func_ = function, .var_ = {var0, var1}};
}

inline space_t UnaryFunctor::operator()(const space_t x) const {
  switch (func_) {
  case UNARY_FUNCTION::IDENTITY:
    return x;

  case UNARY_FUNCTION::MIN:
    return std::fmin(x, var_[0]);

  case UNARY_FUNCTION::MAX:
    return std::fmax(x, var_[0]);

  case UNARY_FUNCTION::LOWER_BOUND:
    return leq_test(x, var_[0]) ? var_[1] : x;

  case UNARY_FUNCTION::UPPER_BOUND:
    return leq_test(var_[0], x) ? var_[1] : x;

  case UNARY_FUNCTION::INVERSE:
    return almost_zero(x) ? std::signbit(x)
                                ? -std::numeric_limits<space_t>::infinity()
                                : std::numeric_limits<space_t>::infinity()
                          : 1. / x;

  case UNARY_FUNCTION::FACTOR:
    return x * var_[0];

  case UNARY_FUNCTION::OFFSET:
    return x + var_[0];

  case UNARY_FUNCTION::EXPONENTIAL:
    return std::exp(x * var_[0]);

  case UNARY_FUNCTION::GAUSSIAN: {
    const auto u = x + var_[0];
    return std::exp(u * u * var_[1]);
  }

  default:
    assert(false && "Invalid unary function name");
    return x;
  }
}

struct DisplacementFunctor {
  DISPLACEMENT_FUNCTION func_ = DISPLACEMENT_FUNCTION::NULL_DF;
  space_t var_ = 0;

  template <Coordinate CoordT>
  space_t operator()(const Displacement<CoordT> &d) const;
};

// Whether the function reads the z component, which only 3D displacements have
constexpr inline bool uses_z_component(const DisplacementFunctor &df) noexcept {
  return df.func_ == DISPLACEMENT_FUNCTION::DISPLACEMENT_Z ||
         df.func_ == DISPLACEMENT_FUNCTION::DISTANCE_Z;
}

inline DisplacementFunctor
construct_displacement_functor(const DISPLACEMENT_FUNCTION function,
                               const space_t var = 0.) {
  return {.func_ = function, .var_ = var};
}

template <Coordinate CoordT>
inline space_t
DisplacementFunctor::operator()(const Displacement<CoordT> &d) const {
  switch (func_) {
  case DISPLACEMENT_FUNCTION::CONSTANT:
    return var_;

  case DISPLACEMENT_FUNCTION::DISTANCE:
    return d.get_distance();

  case DISPLACEMENT_FUNCTION::DISPLACEMENT_X:
    return d.displacement_.x_;

  case DISPLACEMENT_FUNCTION::DISPLACEMENT_Y:
    return d.displacement_.y_;

  case DISPLACEMENT_FUNCTION::DISPLACEMENT_Z: {
    if constexpr (std::is_same_v<CoordT, Coord3D>)
      return d.displacement_.z_;
    else
      assert(false && "Invalid displacement function name");
    return 0;
  }

  case DISPLACEMENT_FUNCTION::DISTANCE_X:
    return std::fabs(d.displacement_.x_);

  case DISPLACEMENT_FUNCTION::DISTANCE_Y:
    return std::fabs(d.displacement_.y_);

  case DISPLACEMENT_FUNCTION::DISTANCE_Z: {
    if constexpr (std::is_same_v<CoordT, Coord3D>)
      return std::fabs(d.displacement_.z_);
    else
      assert(false && "Invalid displacement function name");
    return 0;
  }

  default:
    assert(false && "Invalid displacement function name");
    return 0;
  }
}
} // namespace sapi

#endif
