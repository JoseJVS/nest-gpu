# cmake/CheckIncludeSymbols.cmake
#
# This file is part of NEST GPU.
#
# Copyright (C) 2004 The NEST Initiative
#
# NEST GPU is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# NEST GPU is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NEST GPU.  If not, see <http://www.gnu.org/licenses/>.

# Here we check for all required include headers, types, symbols and functions.

include( CheckIncludeFiles )
check_include_files( "dlfcn.h" HAVE_DLFCN_H )
check_include_files( "inttypes.h" HAVE_INTTYPES_H )
check_include_files( "limits.h" HAVE_LIMITS_H )
check_include_files( "memory.h" HAVE_MEMORY_H )
check_include_files( "stdint.h" HAVE_STDINT_H )
check_include_files( "stdlib.h" HAVE_STDLIB_H )
check_include_files( "strings.h" HAVE_STRINGS_H )
check_include_files( "string.h" HAVE_STRING_H )
check_include_files( "sys/stat.h" HAVE_SYS_STAT_H )
check_include_files( "sys/types.h" HAVE_SYS_TYPES_H )
check_include_files( "unistd.h" HAVE_UNISTD_H )
