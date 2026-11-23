/*
 *  thread_aligned_array.h
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

#ifndef THREAD_ALIGNED_ARRAY_H
#define THREAD_ALIGNED_ARRAY_H

#include <cassert>
#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "sapi_config.h"
#include "sapi_forward_declarations.h"

namespace sapi {
// Forward definition to link with vp_interface
vp_t get_thread_num();
vp_t get_max_omp_threads();

template <typename T>
  requires std::is_copy_constructible_v<T>
class TAArray {
public:
  bool is_initialized() const;

  void clear();
  void prepare();
  void clone(const T &item);

  T *get_thread_item(const vp_t tid) const;
  const T *get_local_thread_item() const;

protected:
  bool cloned_ = false;
  bool prepared_ = false;
  std::vector<std::unique_ptr<T>> item_vec_;
};

template <typename T>
  requires std::is_copy_constructible_v<T>
inline void TAArray<T>::clear() {
  item_vec_.clear();
  cloned_ = false;
  prepared_ = false;
}

template <typename T>
  requires std::is_copy_constructible_v<T>
inline bool TAArray<T>::is_initialized() const {
  assert(cloned_ <= prepared_);
  return cloned_;
}

template <typename T>
  requires std::is_copy_constructible_v<T>
inline void TAArray<T>::prepare() {
  if (!prepared_ || cloned_ ||
      !std::cmp_equal(item_vec_.size(), get_max_omp_threads())) {
    item_vec_.clear();
    assert(0 < get_max_omp_threads());
    item_vec_.resize(static_cast<std::size_t>(get_max_omp_threads()));
    prepared_ = true;
    cloned_ = false;
  }
}

template <typename T>
  requires std::is_copy_constructible_v<T>
inline void TAArray<T>::clone(const T &item) {
  prepare();
  assert(!cloned_ && prepared_);

  // prepare() sized item_vec_ to the maximum thread count
#pragma omp parallel default(none) shared(item_vec_, item)
  item_vec_[static_cast<std::size_t>(get_thread_num())] =
      std::make_unique<T>(item);

  cloned_ = true;
}

template <typename T>
  requires std::is_copy_constructible_v<T>
inline T *TAArray<T>::get_thread_item(const vp_t tid) const {
  // The thread count stays fixed once items are cloned
  assert(0 <= tid && std::cmp_less(tid, item_vec_.size()));
  return item_vec_[static_cast<std::size_t>(tid)].get();
}

template <typename T>
  requires std::is_copy_constructible_v<T>
inline const T *TAArray<T>::get_local_thread_item() const {
  return get_thread_item(get_thread_num());
}
} // namespace sapi

#endif
