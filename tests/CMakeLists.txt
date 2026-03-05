# -------------------------------------------------------GoogleTest MPI smoke tests--
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
# Prevent GoogleTest from overriding our compiler/linker settings.
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(test_multidim_array
    ${CMAKE_SOURCE_DIR}/tests/test_multidim_array.cpp
)
add_dependencies(test_multidim_array relion_lib)
target_compile_definitions(test_multidim_array PRIVATE RELION_UNIT_TESTS)
target_include_directories(test_multidim_array PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${FFTW_INCLUDES}
)
target_link_libraries(test_multidim_array
    GTest::gtest
    relion_lib
    ${FFTW_LIBRARIES}
)

add_executable(test_expectation_mpi
    ${CMAKE_SOURCE_DIR}/tests/test_expectation_mpi.cpp
)
add_dependencies(test_expectation_mpi relion_lib)
target_compile_definitions(test_expectation_mpi PRIVATE RELION_UNIT_TESTS)
target_include_directories(test_expectation_mpi PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${FFTW_INCLUDES}
)
target_link_libraries(test_expectation_mpi
    GTest::gtest
    relion_lib
    MPI::MPI_CXX
    ${FFTW_LIBRARIES}
)
