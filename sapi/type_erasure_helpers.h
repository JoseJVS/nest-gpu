/*
 *  type_erasure_helpers.h
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

#ifndef TYPE_ERASURE_HELPERS_H
#define TYPE_ERASURE_HELPERS_H

#include <concepts>
#include <initializer_list>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "sapi_config.h"
#include "sapi_forward_declarations.h"

namespace sapi {
template <typename RT> struct StateLessCreator {
  virtual ~StateLessCreator() noexcept = default;

  virtual RT create() const;

  virtual RT create(const std::vector<space_t> &) const;

  virtual RT create(const std::vector<space_t> &,
                    const std::vector<angle_t> &) const;

  virtual RT create(const std::vector<space_t> &, const std::vector<space_t> &,
                    const std::vector<space_t> &) const;
};

template <typename RT> inline RT StateLessCreator<RT>::create() const {
  throw std::runtime_error("Undefined creation method for state-less creator");
}

template <typename RT>
inline RT StateLessCreator<RT>::create(const std::vector<space_t> &) const {
  throw std::runtime_error("Undefined creation method for state-less creator");
}

template <typename RT>
inline RT StateLessCreator<RT>::create(const std::vector<space_t> &,
                                       const std::vector<angle_t> &) const {
  throw std::runtime_error("Undefined creation method for state-less creator");
}

template <typename RT>
inline RT StateLessCreator<RT>::create(const std::vector<space_t> &,
                                       const std::vector<space_t> &,
                                       const std::vector<space_t> &) const {
  throw std::runtime_error("Undefined creation method for state-less creator");
}

template <typename RT> struct RNGConcept {
  virtual ~RNGConcept() noexcept = default;
  virtual std::unique_ptr<RNGConcept<RT>> clone() const noexcept = 0;
  virtual void seed(const std::initializer_list<rng_seed_t> &seeds) = 0;
  virtual RT operator()() = 0;
};

template <typename RNG>
  requires std::is_nothrow_copy_constructible_v<RNG>
class RNGModel final : public RNGConcept<typename RNG::result_type> {
public:
  using result_type = typename RNG::result_type;

  RNGModel(RNG) noexcept;

  // Defined in rng_creators.cpp
  void seed(const std::initializer_list<rng_seed_t> &seeds) override;

  result_type operator()() override;

  std::unique_ptr<RNGConcept<result_type>> clone() const noexcept override;

protected:
  RNG engine_;
};

template <typename RNG>
  requires std::is_nothrow_copy_constructible_v<RNG>
RNGModel<RNG>::RNGModel(RNG engine) noexcept : engine_(engine) {}

template <typename RNG>
  requires std::is_nothrow_copy_constructible_v<RNG>
inline typename RNG::result_type RNGModel<RNG>::operator()() {
  return engine_();
}

template <typename RNG>
  requires std::is_nothrow_copy_constructible_v<RNG>
inline std::unique_ptr<RNGConcept<typename RNG::result_type>>
RNGModel<RNG>::clone() const noexcept {
  return std::make_unique<RNGModel>(engine_);
}

template <RNGResult RT> class AnyRNG_T {
public:
  using result_type = RT;

  template <typename RNG>
    requires std::same_as<typename RNG::result_type, RT>
  AnyRNG_T(RNG) noexcept;

  AnyRNG_T() = delete;
  AnyRNG_T(const AnyRNG_T &) noexcept;
  AnyRNG_T(AnyRNG_T &&) = default;
  ~AnyRNG_T() noexcept = default;

  AnyRNG_T &operator=(const AnyRNG_T &) = delete;
  AnyRNG_T &operator=(AnyRNG_T &&) = delete;

  constexpr static RT min();

  constexpr static RT max();

  void seed(const std::initializer_list<rng_seed_t> &seeds);

  RT operator()();

protected:
  const std::unique_ptr<RNGConcept<RT>> rng_;
  RNGConcept<RT> *const ptr_;
};

template <RNGResult RT>
template <typename RNG>
  requires std::same_as<typename RNG::result_type, RT>
AnyRNG_T<RT>::AnyRNG_T(RNG engine) noexcept
    : rng_(std::make_unique<RNGModel<RNG>>(engine)), ptr_(rng_.get()) {}

template <RNGResult RT>
AnyRNG_T<RT>::AnyRNG_T(const AnyRNG_T &any_rng) noexcept
    : rng_(any_rng.ptr_->clone()), ptr_(rng_.get()) {}

template <RNGResult RT> constexpr inline RT AnyRNG_T<RT>::min() { return 0; }

template <RNGResult RT> constexpr inline RT AnyRNG_T<RT>::max() {
  return std::numeric_limits<RT>::max();
}

template <RNGResult RT>
inline void AnyRNG_T<RT>::seed(const std::initializer_list<rng_seed_t> &seeds) {
  ptr_->seed(seeds);
}

template <RNGResult RT> inline RT AnyRNG_T<RT>::operator()() {
  return ptr_->operator()();
}
} // namespace sapi

#endif
