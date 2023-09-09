// Copyright 2022 Philipp.Duernay
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#ifndef VSLAM_MACROS_H__
#define VSLAM_MACROS_H__

#ifdef __GNUC__
#define UNUSED(x) UNUSED_##x __attribute__((__unused__))
#else
#define UNUSED(x) UNUSED_##x
#endif

#ifdef __GNUC__
#define UNUSED_FUNCTION(x) __attribute__((__unused__)) UNUSED_##x
#else
#define UNUSED_FUNCTION(x) UNUSED_##x
#endif

#include <memory>
#include <vector>
#define TYPEDEF_PTR(name) \
  typedef std::shared_ptr<name> ShPtr; \
  typedef std::shared_ptr<const name> ConstShPtr; \
  typedef std::unique_ptr<name> UnPtr; \
  typedef std::unique_ptr<const name> ConstUnPtr; \
  typedef std::vector<ShPtr> VecShPtr; \
  typedef std::vector<ConstShPtr> VecConstShPtr; \
  typedef std::vector<UnPtr> VecUnPtr; \
  typedef std::vector<ConstUnPtr> VecConstUnPtr;

#endif
