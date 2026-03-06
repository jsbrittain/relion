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
add_test(NAME test_multidim_array COMMAND test_multidim_array)

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
add_test(NAME test_expectation_mpi
    COMMAND mpirun -n 2 $<TARGET_FILE:test_expectation_mpi>
)

foreach(test_name IN ITEMS test_matrix1d test_matrix2d test_euler test_strings test_ctf test_metadata_table)
    add_executable(${test_name}
        ${CMAKE_SOURCE_DIR}/tests/${test_name}.cpp
    )
    add_dependencies(${test_name} relion_lib)
    target_compile_definitions(${test_name} PRIVATE RELION_UNIT_TESTS)
    target_include_directories(${test_name} PRIVATE
        ${CMAKE_SOURCE_DIR}
        ${FFTW_INCLUDES}
    )
    target_link_libraries(${test_name}
        GTest::gtest
        relion_lib
        ${FFTW_LIBRARIES}
    )
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

# ----------------------------------------------------------------COVERAGE TARGET--
# Requires: COVERAGE=ON at cmake time, lcov >= 1.14, genhtml (part of lcov).
# Usage:
#   cmake -DCOVERAGE=ON -DBUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Debug ...
#   make -j$(nproc)
#   make coverage          # runs tests, collects .gcda, produces coverage/index.html
if(COVERAGE)
    find_program(LCOV_PATH lcov REQUIRED)
    find_program(GENHTML_PATH genhtml REQUIRED)

    set(COVERAGE_DIR ${CMAKE_BINARY_DIR}/coverage)
    set(COVERAGE_INFO ${COVERAGE_DIR}/coverage.info)
    set(COVERAGE_FILTERED ${COVERAGE_DIR}/coverage_filtered.info)

    add_custom_target(coverage
        # 0. Ensure output directory exists and reset stale counters.
        COMMAND ${CMAKE_COMMAND} -E make_directory ${COVERAGE_DIR}
        COMMAND ${LCOV_PATH} --zerocounters --directory ${CMAKE_BINARY_DIR}

        # 1. Run the non-MPI test binaries.
        COMMAND ${CMAKE_BINARY_DIR}/bin/test_multidim_array
        COMMAND ${CMAKE_BINARY_DIR}/bin/test_matrix1d
        COMMAND ${CMAKE_BINARY_DIR}/bin/test_matrix2d
        COMMAND ${CMAKE_BINARY_DIR}/bin/test_euler
        COMMAND ${CMAKE_BINARY_DIR}/bin/test_strings
        COMMAND ${CMAKE_BINARY_DIR}/bin/test_ctf
        COMMAND ${CMAKE_BINARY_DIR}/bin/test_metadata_table

        # 2. Run the MPI test binary (requires mpirun; adjust -n as needed).
        COMMAND mpirun -n 2 ${CMAKE_BINARY_DIR}/bin/test_expectation_mpi

        # 3. Collect coverage counters into a single .info file.
        COMMAND ${LCOV_PATH}
            --capture
            --directory ${CMAKE_BINARY_DIR}
            --output-file ${COVERAGE_INFO}
            --ignore-errors mismatch,gcov

        # 4. Strip uninteresting paths: system headers, googletest internals,
        #    and the tests themselves (we want library coverage, not test coverage).
        COMMAND ${LCOV_PATH}
            --remove ${COVERAGE_INFO}
            "/usr/*"
            "${CMAKE_BINARY_DIR}/_deps/*"
            "${CMAKE_SOURCE_DIR}/tests/*"
            --output-file ${COVERAGE_FILTERED}

        # 5. Render HTML report.
        COMMAND ${GENHTML_PATH}
            --title "RELION coverage"
            --legend
            --output-directory ${COVERAGE_DIR}
            ${COVERAGE_FILTERED}

        COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --cyan
            "Coverage report: ${COVERAGE_DIR}/index.html"

        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Collecting gcov data and generating HTML coverage report"
        VERBATIM
    )
endif()
