cmake_minimum_required (VERSION 3.21.0)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-ffast-math")

option(BUILD_SAMPLE "Build the MNIST sample network (requires OpenCV)" ON)

option(BUILD_UNIT_TESTS "Build library unit tests" ON)
option(BUILD_INTEGRATION_TESTS "Build library integration tests" ON)
option(BUILD_SYSTEM_TESTS "Build library system tests" OFF)

option(BUILD_BENCH "Build library benchmarks" ON)

project(nn-cpp)

add_subdirectory(nn-cpp)

if (BUILD_SAMPLE)
    add_subdirectory(sample)
endif()

if (BUILD_UNIT_TESTS OR BUILD_INTEGRATION_TESTS OR BUILD_SYSTEM_TESTS)
    add_subdirectory(test)
endif()

if (BUILD_SYSTEM_TESTS)
    set(URLS
        "https://www.kaggle.com/api/v1/datasets/download/samihatna/fashion-mnist"
    )

    set(DEST_DIR "${CMAKE_BINARY_DIR}/test")
    file(MAKE_DIRECTORY ${DEST_DIR})

    find_program(UNZIP_TOOL unzip)
    if(NOT UNZIP_TOOL)
        message(FATAL_ERROR "Unzip tool not found. Please install 'unzip' and try again.")
    endif()

    foreach(URL ${URLS})
        get_filename_component(FILENAME ${URL} NAME_WE)
        set(ARCHIVE_ZIP "${CMAKE_BINARY_DIR}/${FILENAME}.zip")
        set(UNZIPPED_DIR "${DEST_DIR}/${FILENAME}")

        file(DOWNLOAD
            ${URL}
            ${ARCHIVE_ZIP}
            SHOW_PROGRESS
            STATUS DOWNLOAD_STATUS
        )

        if(NOT EXISTS ${UNZIPPED_DIR})
            execute_process(
                COMMAND ${UNZIP_TOOL} ${ARCHIVE_ZIP} -d ${UNZIPPED_DIR}
                RESULT_VARIABLE UNZIP_RESULT
                OUTPUT_QUIET
                ERROR_QUIET
            )
            if(NOT UNZIP_RESULT EQUAL 0)
                message(FATAL_ERROR "Failed to unzip ${ARCHIVE_ZIP}. Please check the file or unzip tool.")
            endif()
        endif()
    endforeach()
endif()


if (BUILD_BENCH)
    add_subdirectory(bench)
endif()
