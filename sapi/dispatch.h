/*
 *  dispatch.h
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

#ifndef DISPATCH_H
#define DISPATCH_H

#include <utility>

namespace sapi {
// Turns runtime bools into template arguments: calls
// f.template operator()<b0, b1, ...>() with the values of the bools, in order.
// Every combination is instantiated, f is usually a template lambda.
template <bool... Bs, typename F> decltype(auto) dispatch_bools(F &&f) {
  return std::forward<F>(f).template operator()<Bs...>();
}

template <bool... Bs, typename F, typename... Rest>
decltype(auto) dispatch_bools(F &&f, const bool b, const Rest... rest) {
  return b ? dispatch_bools<Bs..., true>(std::forward<F>(f), rest...)
           : dispatch_bools<Bs..., false>(std::forward<F>(f), rest...);
}
} // namespace sapi

#endif
