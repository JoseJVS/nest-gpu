# sapi/CMakeLists.txt
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

set ( sapi_kernel_sources
tiles2d.h
node_slicing.h
api_converters.h
mask_containers.h
grid_generation.h
sapi_config.h
connection_containers.h
creator_registry.h
c_api.cpp
mask_geometry.h
grid_functors2d.h
connection_generators.h
bounding_box.h
node_distribution.h
connection_functor.h
grid_containers.h
spatial_manager.h
gf_creators.h
numerics.h
node_insertion.h
node_creation.h
grid_functors.h
node_containers.h
numeric_functors.h
timer_register.h
nf_creators.h
cg_creators.h
spatial_communication.h
thread_aligned_array.h
type_erasure_helpers.h
grid_neighborhood.h
mask_collection.h
masks2d.h
mask_creators.h
tile.h
random_manager.h
tile_grid.h
garbage_collector.h
thirdparty/pcg_extras.hpp
thirdparty/pcg_random.hpp
thirdparty/random_generators.h
thirdparty/randutils.hpp
connection_generation.h
node_collection.h
coordinates.h
gf_collection.h
api_containers.h
mask_node_processing.h
vp_interface.h
mask.h
coordinate_geometry.h
spatial_containers.h
mask_tile_processing.h
c_api.h
)

add_library( sapi_kernel STATIC ${sapi_kernel_sources} )

target_link_libraries( sapi_kernel PUBLIC
    ${MPI_CXX_LIBRARIES}
    m
    )

target_include_directories( sapi_kernel PRIVATE
    ${PROJECT_SOURCE_DIR}/sapi
    ${PROJECT_SOURCE_DIR}/sapi/thirdparty
    ${PROJECT_BINARY_DIR}/libnestutil
    )

FILTER_HEADERS( "${sapi_kernel_sources}" install_headers )
install( FILES ${install_headers}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/nestgpu )
