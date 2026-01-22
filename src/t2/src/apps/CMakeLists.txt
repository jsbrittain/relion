# Prefix to avoid build collisions
if(NOT DEFINED APPS_PREFIX)
    set(APPS_PREFIX "t2_")
endif()

set(RELION_LIB          "${APPS_PREFIX}relion_lib")
set(RELION_GPU_UTIL     "${APPS_PREFIX}relion_gpu_util")
set(RELION_JAZ_GPU_UTIL "${APPS_PREFIX}relion_jaz_gpu_util")
set(RELION_SYCL_UTIL    "${APPS_PREFIX}relion_sycl_util")

set(T2_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../..")
include_directories("${T2_SOURCE_DIR}")

message("Running apps/CMakeLists.txt...")

if(NOT MKLFFT)
	include_directories("${FFTW_INCLUDES}")
endif(NOT MKLFFT)
#include_directories(${CMAKE_BINARY_DIR}/include)

find_path (X11_INCLUDES Xdbe.h)

message(STATUS "CMAKE_BINARY_DIR:" ${CMAKE_BINARY_DIR})

# AUTOMATIC VERSIONING
# Reference: https://stackoverflow.com/questions/1435953/how-can-i-pass-git-sha1-to-compiler-as-definition-using-cmake

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")
include(GetGitRevisionDescription)
get_git_head_revision(GIT_REFSPEC GIT_SHA1_FULL)
set(RELION_VERSION_SUFFIX "")
if (GIT_SHA1_FULL)
   string(SUBSTRING ${GIT_SHA1_FULL} 0 6 GIT_SHA1_PREFIX)
   if (NOT GIT_SHA1_PREFIX STREQUAL "GITDIR")
      message(STATUS "Git commit ID: ${GIT_SHA1_FULL}")
      set(RELION_VERSION_SUFFIX "-commit-${GIT_SHA1_PREFIX}")
   endif()
endif()
configure_file("${T2_SOURCE_DIR}/src/macros.cpp.in" "${CMAKE_BINARY_DIR}/macros.cpp" @ONLY)

if (ALTCPU)
	file(GLOB REL_SRC "${T2_SOURCE_DIR}/src/*.cpp" "${CMAKE_BINARY_DIR}/macros.cpp" "${T2_SOURCE_DIR}/src/*.c" "${T2_SOURCE_DIR}/src/acc/cpu/*.cpp" "${T2_SOURCE_DIR}/src/acc/cpu/cpu_kernels/*.cpp" )
	file(GLOB REL_SRC_H "${T2_SOURCE_DIR}/src/*.h" "${T2_SOURCE_DIR}/src/acc/*.h" "${T2_SOURCE_DIR}/src/acc/cpu/*.h" "${T2_SOURCE_DIR}/src/acc/cpu/cpu_kernels/*.h" )
else()
	file(GLOB REL_SRC "${T2_SOURCE_DIR}/src/*.cpp" "${CMAKE_BINARY_DIR}/macros.cpp" "${T2_SOURCE_DIR}/src/*.c" "${T2_SOURCE_DIR}/src/acc/*.cpp" )
	file(GLOB REL_SRC_H "${T2_SOURCE_DIR}/src/*.h" "${T2_SOURCE_DIR}/src/acc/*.h" ) 
endif(ALTCPU)

file(GLOB REL_JAZ_MATH_SRC "${T2_SOURCE_DIR}/src/jaz/math/*.cpp")
file(GLOB REL_JAZ_MATH_H "${T2_SOURCE_DIR}/src/jaz/math/*.h")
file(GLOB REL_JAZ_IMG_PROC_SRC "${T2_SOURCE_DIR}/src/jaz/image/*.cpp")
file(GLOB REL_JAZ_IMG_PROC_H "${T2_SOURCE_DIR}/src/jaz/image/*.h")
file(GLOB REL_JAZ_MESH_PROC_SRC "${T2_SOURCE_DIR}/src/jaz/mesh/*.cpp")
file(GLOB REL_JAZ_MESH_PROC_H "${T2_SOURCE_DIR}/src/jaz/mesh/*.h")
file(GLOB REL_JAZ_UTIL_SRC "${T2_SOURCE_DIR}/src/jaz/util/*.cpp")
file(GLOB REL_JAZ_UTIL_H "${T2_SOURCE_DIR}/src/jaz/util/*.h")
file(GLOB REL_JAZ_TOMO_SRC "${T2_SOURCE_DIR}/src/jaz/tomography/*.cpp")
file(GLOB REL_JAZ_TOMO_H "${T2_SOURCE_DIR}/src/jaz/tomography/*.h")
file(GLOB REL_JAZ_TOMO_PROJ_SRC "${T2_SOURCE_DIR}/src/jaz/tomography/projection/*.cpp")
file(GLOB REL_JAZ_TOMO_PROJ_H "${T2_SOURCE_DIR}/src/jaz/tomography/projection/*.h")
file(GLOB REL_JAZ_TOMO_LATTICE_SRC "${T2_SOURCE_DIR}/src/jaz/tomography/lattice/*.cpp")
file(GLOB REL_JAZ_TOMO_LATTICE_H "${T2_SOURCE_DIR}/src/jaz/tomography/lattice/*.h")
file(GLOB REL_JAZ_TOMO_MEMBRANE_SRC "${T2_SOURCE_DIR}/src/jaz/membrane/*.cpp")
file(GLOB REL_JAZ_TOMO_MEMBRANE_H "${T2_SOURCE_DIR}/src/jaz/membrane/*.h")
file(GLOB REL_JAZ_TOMO_FILAMENT_SRC "${T2_SOURCE_DIR}/src/jaz/tomography/filament/*.cpp")
file(GLOB REL_JAZ_TOMO_FILAMENT_H "${T2_SOURCE_DIR}/src/jaz/tomography/filament/*.h")
file(GLOB REL_JAZ_TOMO_MOTION_SRC "${T2_SOURCE_DIR}/src/jaz/tomography/motion/*.cpp")
file(GLOB REL_JAZ_TOMO_MOTION_H "${T2_SOURCE_DIR}/src/jaz/tomography/motion/*.h")
file(GLOB REL_JAZ_TOMO_MOTION_MODULAR_ALIGNMENT_SRC "${T2_SOURCE_DIR}/src/jaz/tomography/motion/modular_alignment/*.cpp")
file(GLOB REL_JAZ_TOMO_MOTION_MODULAR_ALIGNMENT_H "${T2_SOURCE_DIR}/src/jaz/tomography/motion/modular_alignment/*.h")
file(GLOB REL_JAZ_TOMO_MANIFOLD_SRC "${T2_SOURCE_DIR}/src/jaz/tomography/manifold/*.cpp")
file(GLOB REL_JAZ_TOMO_MANIFOLD_H "${T2_SOURCE_DIR}/src/jaz/tomography/manifold/*.h")
file(GLOB REL_JAZ_OPTIM_SRC "${T2_SOURCE_DIR}/src/jaz/optimization/*.cpp")
file(GLOB REL_JAZ_OPTIM_H "${T2_SOURCE_DIR}/src/jaz/optimization/*.h")
file(GLOB REL_JAZ_OPTICS_SRC "${T2_SOURCE_DIR}/src/jaz/optics/*.cpp")
file(GLOB REL_JAZ_OPTICS_H "${T2_SOURCE_DIR}/src/jaz/optics/*.h")
file(GLOB REL_JAZ_OPTICS_DUAL_CONTRAST_SRC "${T2_SOURCE_DIR}/src/jaz/optics/dual_contrast/*.cpp")
file(GLOB REL_JAZ_OPTICS_DUAL_CONTRAST_H "${T2_SOURCE_DIR}/src/jaz/optics/dual_contrast/*.h")
file(GLOB REL_JAZ_SEGMENTATION_SRC "${T2_SOURCE_DIR}/src/jaz/segmentation/*.cpp")
file(GLOB REL_JAZ_SEGMENTATION_H "${T2_SOURCE_DIR}/src/jaz/segmentation/*.h")
file(GLOB REL_JAZ_ATOMIC_SRC "${T2_SOURCE_DIR}/src/jaz/atomic/*.cpp")
file(GLOB REL_JAZ_ATOMIC_H "${T2_SOURCE_DIR}/src/jaz/atomic/*.h")
file(GLOB REL_JAZ_PROGRAMS_SRC "${T2_SOURCE_DIR}/src/jaz/tomography/programs/*.cpp")
file(GLOB REL_JAZ_PROGRAMS_H "${T2_SOURCE_DIR}/src/jaz/tomography/programs/*.h")
file(GLOB REL_JAZ_DYNAMO_SRC "${T2_SOURCE_DIR}/src/jaz/tomography/dynamo/*.cpp")
file(GLOB REL_JAZ_DYNAMO_H "${T2_SOURCE_DIR}/src/jaz/tomography/dynamo/*.h")

file(GLOB REL_JAZ_SPA_SRC "${T2_SOURCE_DIR}/src/jaz/single_particle/*.cpp")
file(GLOB REL_JAZ_SPA_H "${T2_SOURCE_DIR}/src/jaz/single_particle/*.h")
file(GLOB REL_JAZ_SPA_OPT_SRC "${T2_SOURCE_DIR}/src/jaz/single_particle/optimization/*.cpp")
file(GLOB REL_JAZ_SPA_OPT_H "${T2_SOURCE_DIR}/src/jaz/single_particle/optimization/*.h")
file(GLOB REL_JAZ_SPA_CTF_SRC "${T2_SOURCE_DIR}/src/jaz/single_particle/ctf/*.cpp")
file(GLOB REL_JAZ_SPA_CTF_H "${T2_SOURCE_DIR}/src/jaz/single_particle/ctf/*.h")
file(GLOB REL_JAZ_SPA_MOTION_SRC "${T2_SOURCE_DIR}/src/jaz/single_particle/motion/*.cpp")
file(GLOB REL_JAZ_SPA_MOTION_H "${T2_SOURCE_DIR}/src/jaz/single_particle/motion/*.h")
file(GLOB REL_JAZ_SPA_MATH_SRC "${T2_SOURCE_DIR}/src/jaz/single_particle/math/*.cpp")
file(GLOB REL_JAZ_SPA_MATH_H "${T2_SOURCE_DIR}/src/jaz/single_particle/math/*.h")
file(GLOB REL_JAZ_SPA_IMG_PROC_SRC "${T2_SOURCE_DIR}/src/jaz/single_particle/img_proc/*.cpp")
file(GLOB REL_JAZ_SPA_IMG_PROC_H "${T2_SOURCE_DIR}/src/jaz/single_particle/img_proc/*.h")
file(GLOB REL_JAZ_SPA_TOMO_SRC "${T2_SOURCE_DIR}/src/jaz/single_particle/tomo/*.cpp")
file(GLOB REL_JAZ_SPA_TOMO_H "${T2_SOURCE_DIR}/src/jaz/single_particle/tomo/*.h")
file(GLOB REL_JAZ_SPA_IO_SRC "${T2_SOURCE_DIR}/src/jaz/single_particle/io/*.cpp")
file(GLOB REL_JAZ_SPA_IO_H "${T2_SOURCE_DIR}/src/jaz/single_particle/io/*.h")
file(GLOB REL_JAZ_SPA_PROGRAMS_SRC "${T2_SOURCE_DIR}/src/jaz/single_particle/programs/*.cpp")
file(GLOB REL_JAZ_SPA_PROGRAMS_H "${T2_SOURCE_DIR}/src/jaz/single_particle/programs/*.h")

file(GLOB REL_SPHERICAL_HARMONICS_SRC "${T2_SOURCE_DIR}/src/spherical-harmonics/*.cc")
file(GLOB REL_SPHERICAL_HARMONICS_H "${T2_SOURCE_DIR}/src/spherical-harmonics/*.h")
file(GLOB REL_D3x3_SRC "${T2_SOURCE_DIR}/src/jaz/d3x3/*.c")
file(GLOB REL_D3x3_H "${T2_SOURCE_DIR}/src/jaz/d3x3/*.h")
file(GLOB REL_LBFGS_SRC "${T2_SOURCE_DIR}/src/lbfgs/*.c")
file(GLOB REL_LBFGS_H "${T2_SOURCE_DIR}/src/lbfgs/*.h")

file(GLOB REL_HP "${T2_SOURCE_DIR}/src/Healpix_2.15a/*.cc")
file(GLOB RELION_TARGETS "${T2_SOURCE_DIR}/src/apps/refine*.cpp")
file(GLOB RELION_JAZ_SPA_TARGETS "${T2_SOURCE_DIR}/src/jaz/single_particle/apps/*.cpp")
file(GLOB RELION_JAZ_TOMO_TARGETS "${T2_SOURCE_DIR}/src/jaz/tomography/apps/*.cpp")


if(RELION_TEST)
	file(GLOB RELION_JAZ_EXPERIMENTAL_TARGETS "${T2_SOURCE_DIR}/src/jaz/scripts/*.cpp")
endif(RELION_TEST)

set(TORCH_TARGETS class_ranker)

#--Remove apps for testing--

set(TEST_TARGETS)
if(NOT RELION_TEST)
	foreach(TARGET ${TEST_TARGETS})
		list(REMOVE_ITEM RELION_TARGETS "${T2_SOURCE_DIR}/src/apps/${TARGET}.cpp")
	endforeach()
endif(NOT RELION_TEST)

# ${RELION_LIB} is STATIC or SHARED type based on BUILD_SHARED_LIBS=ON/OFF
# ${RELION_LIB} only contains non-X11 parts
if(BUILD_SHARED_LIBS)
	add_library(${RELION_LIB} SHARED
		${REL_SRC} ${REL_SRC_H} ${REL_HP}
		${REL_JAZ_MATH_SRC} ${REL_JAZ_MATH_H}
		${REL_JAZ_IMG_PROC_SRC} ${REL_JAZ_IMG_PROC_H}
		${REL_JAZ_MESH_PROC_SRC} ${REL_JAZ_MESH_PROC_H}
		${REL_JAZ_UTIL_SRC} ${REL_JAZ_UTIL_H}
		${REL_JAZ_TOMO_SRC} ${REL_JAZ_TOMO_H}
		${REL_JAZ_TOMO_PROJ_SRC} ${REL_JAZ_TOMO_PROJ_H}
		${REL_JAZ_TOMO_MEMBRANE_SRC} ${REL_JAZ_TOMO_MEMBRANE_H}
		${REL_JAZ_TOMO_FILAMENT_SRC} ${REL_JAZ_TOMO_FILAMENT_H}
		${REL_JAZ_TOMO_MOTION_SRC} ${REL_JAZ_TOMO_MOTION_H}
		${REL_JAZ_TOMO_MOTION_MODULAR_ALIGNMENT_SRC} ${REL_JAZ_TOMO_MOTION_MODULAR_ALIGNMENT_H}
		${REL_JAZ_TOMO_MANIFOLD_SRC} ${REL_JAZ_TOMO_MANIFOLD_H}
		${REL_JAZ_TOMO_LATTICE_SRC} ${REL_JAZ_TOMO_LATTICE_H}
		${REL_JAZ_OPTIM_SRC} ${REL_JAZ_OPTIM_H}
		${REL_JAZ_OPTICS_SRC} ${REL_JAZ_OPTICS_H}
		${REL_JAZ_OPTICS_DUAL_CONTRAST_SRC} ${REL_JAZ_OPTICS_DUAL_CONTRAST_H}
		${REL_JAZ_SEGMENTATION_SRC} ${REL_JAZ_SEGMENTATION_H}
		${REL_JAZ_ATOMIC_SRC} ${REL_JAZ_ATOMIC_H}
		${REL_JAZ_PROGRAMS_SRC} ${REL_JAZ_PROGRAMS_H}
		${REL_JAZ_DYNAMO_SRC} ${REL_JAZ_DYNAMO_H}
		${REL_JAZ_SPA_SRC} ${REL_JAZ_SPA_H}
		${REL_JAZ_SPA_OPT_SRC} ${REL_JAZ_SPA_OPT_H}
		${REL_JAZ_SPA_CTF_SRC} ${REL_JAZ_SPA_CTF_H}
		${REL_JAZ_SPA_MOTION_SRC} ${REL_JAZ_SPA_MOTION_H}
		${REL_JAZ_SPA_MATH_SRC} ${REL_JAZ_SPA_MATH_H}
		${REL_JAZ_SPA_IMG_PROC_SRC} ${REL_JAZ_SPA_IMG_PROC_H}
		${REL_JAZ_SPA_TOMO_SRC} ${REL_JAZ_SPA_TOMO_H}
		${REL_JAZ_SPA_IO_SRC} ${REL_JAZ_SPA_IO_H}
		${REL_JAZ_SPA_PROGRAMS_SRC} ${REL_JAZ_SPA_PROGRAMS_H}
		${REL_D3x3_SRC} ${REL_D3x3_H}
		${REL_LBFGS_SRC} ${REL_LBFGS_H}
		${REL_SPHERICAL_HARMONICS_SRC} ${REL_SPHERICAL_HARMONICS_H})

	install(TARGETS ${RELION_LIB} LIBRARY DESTINATION lib)
else()
	add_library(${RELION_LIB} STATIC
		${REL_SRC} ${REL_SRC_H} ${REL_HP}
		${REL_JAZ_MATH_SRC} ${REL_JAZ_MATH_H}
		${REL_JAZ_IMG_PROC_SRC} ${REL_JAZ_IMG_PROC_H}
		${REL_JAZ_MESH_PROC_SRC} ${REL_JAZ_MESH_PROC_H}
		${REL_JAZ_UTIL_SRC} ${REL_JAZ_UTIL_H}
		${REL_JAZ_TOMO_SRC} ${REL_JAZ_TOMO_H}
		${REL_JAZ_TOMO_PROJ_SRC} ${REL_JAZ_TOMO_PROJ_H}    
		${REL_JAZ_TOMO_MEMBRANE_SRC} ${REL_JAZ_TOMO_MEMBRANE_H}
		${REL_JAZ_TOMO_FILAMENT_SRC} ${REL_JAZ_TOMO_FILAMENT_H}
		${REL_JAZ_TOMO_MOTION_SRC} ${REL_JAZ_TOMO_MOTION_H}
		${REL_JAZ_TOMO_MOTION_MODULAR_ALIGNMENT_SRC} ${REL_JAZ_TOMO_MOTION_MODULAR_ALIGNMENT_H}
		${REL_JAZ_TOMO_MANIFOLD_SRC} ${REL_JAZ_TOMO_MANIFOLD_H}
		${REL_JAZ_TOMO_LATTICE_SRC} ${REL_JAZ_TOMO_LATTICE_H}    
		${REL_JAZ_OPTIM_SRC} ${REL_JAZ_OPTIM_H}
		${REL_JAZ_OPTICS_SRC} ${REL_JAZ_OPTICS_H}
		${REL_JAZ_OPTICS_DUAL_CONTRAST_SRC} ${REL_JAZ_OPTICS_DUAL_CONTRAST_H}
		${REL_JAZ_SEGMENTATION_SRC} ${REL_JAZ_SEGMENTATION_H} 
		${REL_JAZ_ATOMIC_SRC} ${REL_JAZ_ATOMIC_H} 
		${REL_JAZ_PROGRAMS_SRC} ${REL_JAZ_PROGRAMS_H} 
		${REL_JAZ_DYNAMO_SRC} ${REL_JAZ_DYNAMO_H} 
		${REL_JAZ_SPA_SRC} ${REL_JAZ_SPA_H}
		${REL_JAZ_SPA_OPT_SRC} ${REL_JAZ_SPA_OPT_H}
		${REL_JAZ_SPA_CTF_SRC} ${REL_JAZ_SPA_CTF_H}
		${REL_JAZ_SPA_MOTION_SRC} ${REL_JAZ_SPA_MOTION_H}
		${REL_JAZ_SPA_MATH_SRC} ${REL_JAZ_SPA_MATH_H}
		${REL_JAZ_SPA_IMG_PROC_SRC} ${REL_JAZ_SPA_IMG_PROC_H}
		${REL_JAZ_SPA_TOMO_SRC} ${REL_JAZ_SPA_TOMO_H}
		${REL_JAZ_SPA_IO_SRC} ${REL_JAZ_SPA_IO_H}
		${REL_JAZ_SPA_PROGRAMS_SRC} ${REL_JAZ_SPA_PROGRAMS_H}
		${REL_D3x3_SRC} ${REL_D3x3_H}
		${REL_LBFGS_SRC} ${REL_LBFGS_H} 
		${REL_SPHERICAL_HARMONICS_SRC} ${REL_SPHERICAL_HARMONICS_H})
endif()

if(NOT MKLFFT)
	target_link_libraries(${RELION_LIB} ${FFTW_LIBRARIES})
	if(BUILD_OWN_FFTW)
		add_dependencies(${RELION_LIB} own_fftw_lib)
	endif()
	if(BUILD_OWN_FFTWF)
		add_dependencies(${RELION_LIB} own_fftwf_lib)
	endif()
endif(NOT MKLFFT)

if (CUDA_FOUND)
	file(GLOB REL_CUDA_SRC "${T2_SOURCE_DIR}/src/acc/cuda/*.cu" "${T2_SOURCE_DIR}/src/acc/cuda/cuda_kernels/*.cu" )
	cuda_add_library(${RELION_GPU_UTIL} ${REL_CUDA_SRC})

	if (${CMAKE_BUILD_TYPE_LOWER} STREQUAL "profiling")
		find_library(NV_TOOLS_LIBRARIES   NAMES nvToolsExt  PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
		list(APPEND EXTRA_LIBS "${NV_TOOLS_LIBRARIES}")
		target_link_libraries(${RELION_GPU_UTIL} ${NV_TOOLS_LIBRARIES})
		message(STATUS "Adding extra library for NVIDIA profiling: ${NV_TOOLS_LIBRARIES}")
	endif()


	# Presently we have a number of (bad) circular dependencies between the gpu util
	# and relion libraries, which cause errors at least on OS X with clang. Tell the
	# compiler to ignore them.
	if(APPLE)
		set(new_link_flags "-undefined suppress -flat_namespace")
		get_target_property(existing_link_flags ${RELION_GPU_UTIL} LINK_FLAGS)
		if(existing_link_flags)
			set(new_link_flags "${existing_link_flags} ${new_link_flags}")
		endif()
		set_target_properties(${RELION_GPU_UTIL} PROPERTIES LINK_FLAGS "${new_link_flags}")
	endif()

	list(APPEND EXTRA_LIBS "${CUDA_CUFFT_LIBRARIES}")
	if(BUILD_SHARED_LIBS)
		install (TARGETS ${RELION_GPU_UTIL} LIBRARY DESTINATION lib)
	else()
		target_link_libraries(${RELION_GPU_UTIL} ${RELION_LIB})
		target_link_libraries(${RELION_GPU_UTIL} ${CUDA_CUFFT_LIBRARIES})
	endif()

	target_link_libraries(${RELION_LIB} ${RELION_GPU_UTIL} ${CUDA_CUFFT_LIBRARIES})
	target_link_libraries(${RELION_LIB} ${RELION_GPU_UTIL} ${CUDA_CUFFT_LIBRARIES} ${CUDA_curand_LIBRARY})



	file(GLOB REL_JAZ_CUDA_SRC "${T2_SOURCE_DIR}/src/jaz/cuda/*.cu" "${T2_SOURCE_DIR}/src/jaz/cuda/kernels/*.cu" )
	cuda_add_library(${RELION_JAZ_GPU_UTIL} ${REL_JAZ_CUDA_SRC})

	#list(APPEND EXTRA_LIBS "${CUDA_CUFFT_LIBRARIES}")
	#if(BUILD_SHARED_LIBS)
	#	install (TARGETS ${RELION_JAZ_GPU_UTIL} LIBRARY DESTINATION lib)
	#else()
	#	target_link_libraries(${RELION_JAZ_GPU_UTIL} ${RELION_LIB})
	#	target_link_libraries(${RELION_JAZ_GPU_UTIL} ${CUDA_CUFFT_LIBRARIES})
	#endif()

	target_link_libraries(${RELION_LIB} ${RELION_JAZ_GPU_UTIL} ${CUDA_CUFFT_LIBRARIES})
	target_link_libraries(${RELION_LIB} ${RELION_JAZ_GPU_UTIL} ${CUDA_CUFFT_LIBRARIES} ${CUDA_curand_LIBRARY})


endif(CUDA_FOUND)

if(TIFF_FOUND)
	#message("TIFF FOUND")
	include_directories(${TIFF_INCLUDE_DIRS})
	target_link_libraries(${RELION_LIB} ${TIFF_LIBRARIES})
else()
	#message("TIFF NOT FOUND")
endif()

find_package(ZLIB)

find_package(PNG)
if(PNG_FOUND)
	add_definitions(-DHAVE_PNG)
	message("PNG FOUND")
	include_directories(${PNG_INCLUDE_DIRS})
	target_link_libraries(${RELION_LIB} ${PNG_LIBRARY})
else()
	message("PNG NOT FOUND")
endif()

find_package(JPEG)
if(JPEG_FOUND)
	add_definitions(-DHAVE_JPEG)
	message("JPEG FOUND")
	include_directories(${JPEG_INCLUDE_DIRS})
	target_link_libraries(${RELION_LIB} ${JPEG_LIBRARY})
else()
	message("JPEG NOT FOUND")
endif()

# if(xjlEigen)
# 	target_link_libraries(${RELION_LIB} Eigen3::Eigen)
# endif()

if(BUILD_OWN_TBB)
	add_dependencies(${RELION_LIB} OWN_TBB)
endif()

foreach (_target ${RELION_TARGETS})
	GET_FILENAME_COMPONENT(_target "relion_${_target}" NAME_WE)  #specify target name WE=WithoutExtension

	if(${_target} STREQUAL "maingui") # We no longer want "relion_maingui"
		set(_target "relion")
		add_executable(${_target} maingui.cpp)
	else()
		add_executable(${_target} ${_target}.cpp )
		set_target_properties(${_target} PROPERTIES PREFIX "relion_")
	endif()

	set(LIB ${RELION_LIB})
	
 	add_dependencies(${_target} ${RELION_LIB})

	if(NOT MKLFFT)
		target_link_libraries(${_target} ${LIB} ${EXTRA_LIBS} ${MPI_LIBRARIES} ${CMAKE_DL_LIBS})
	else()	
		target_link_libraries(${_target} ${LIB} ${FFTW_LIBRARIES} ${EXTRA_LIBS} ${MPI_LIBRARIES} ${CMAKE_DL_LIBS})
	endif(NOT MKLFFT)

	if(CUDA_FOUND)
		target_link_libraries(${_target} ${RELION_GPU_UTIL})
	endif(CUDA_FOUND)
	
	if (ALTCPU)
		target_link_libraries(${_target} ${TBB_LIBRARIES})
	endif(ALTCPU)

	set_target_properties(${_target} PROPERTIES
	                      XX_STANDARD 11
	                      XX_STANDARD_REQUIRED ON
	                      XX_EXTENSIONS OFF
	)
	
	if(TIFF_FOUND)
		target_link_libraries(${_target} ${TIFF_LIBRARIES})
	endif()
    
	install (TARGETS ${_target} RUNTIME DESTINATION bin)

endforeach()

FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
	#message("OPENMP FOUND")
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
	target_link_libraries(${RELION_LIB} ${OpenMP_omp_LIBRARY})
endif()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
