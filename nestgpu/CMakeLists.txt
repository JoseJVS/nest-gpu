# nest/CMakeLists.txt
#
# This file is part of NEST.
#
# Copyright (C) 2004 The NEST Initiative
#
# NEST is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# NEST is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NEST.  If not, see <http://www.gnu.org/licenses/>.

set( nestgpu_lib_sources
    nestgpu_C.cpp nestgpu_C.h
    )

add_library( nestgpu_lib SHARED ${nestgpu_lib_sources} )
set_target_properties( nestgpu_lib
    PROPERTIES
    VERSION ${NEST_GPU_VERSION}
    SOVERSION 2
    )

target_link_libraries( nestgpu_lib PUBLIC nestgpu_kernel sapi_kernel OpenMP::OpenMP_CXX )

target_compile_options( nestgpu_lib PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:$<TARGET_PROPERTY:OpenMP::OpenMP_CXX,INTERFACE_COMPILE_OPTIONS>>
    $<$<COMPILE_LANGUAGE:CUDA>:$<TARGET_PROPERTY:OpenMP::OpenMP_CXX,INTERFACE_COMPILE_OPTIONS>>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler,$<TARGET_PROPERTY:OpenMP::OpenMP_CXX,INTERFACE_COMPILE_OPTIONS>>
)

if( with-warning )
    target_compile_options( nestgpu_lib
        PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options ${CUDA_WARNING_FLAGS}>
    )
endif()

target_include_directories( nestgpu_lib PRIVATE
    ${PROJECT_SOURCE_DIR}/nestgpu
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/sapi
    ${PROJECT_SOURCE_DIR}/sapi/thirdparty
    ${PROJECT_BINARY_DIR}/libnestutil
    )

install( TARGETS nestgpu_lib
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/nestgpu
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}/nestgpu
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

FILTER_HEADERS( "${nestgpu_lib_sources}" install_headers )
install( FILES ${install_headers}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/nestgpu )
