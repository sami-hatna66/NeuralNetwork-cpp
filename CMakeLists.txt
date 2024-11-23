cmake_minimum_required (VERSION 3.21.0)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-ffast-math")

option(BUILD_SAMPLE "Build the MNIST sample network (requires OpenCV)" ON)

option(BUILD_TESTS "Build library tests" ON)

option(BUILD_BENCH "Build library benchmarks" ON)

project(nn-cpp)

add_subdirectory(nn-cpp)

if (BUILD_SAMPLE)
    add_subdirectory(sample)
endif()

if (BUILD_TESTS)
    add_subdirectory(test)

    set(URLS
    "https://www.kaggle.com/api/v1/datasets/download/samihatna/fashion-mnist"
    "https://www.kaggle.com/api/v1/datasets/download/samihatna/fashion-mnist-system-test"
)

    set(DEST_DIR "${CMAKE_BINARY_DIR}/test")
    file(MAKE_DIRECTORY ${DEST_DIR})

    find_program(UNZIP_TOOL unzip)
    if(NOT UNZIP_TOOL)
        message(FATAL_ERROR "Unzip tool not found. Please install 'unzip' and try again.")
    endif()

    set(ARCHIVE_ZIPS "")
    set(UNZIP_TARGETS "")

    foreach(URL ${URLS})
        get_filename_component(FILENAME ${URL} NAME_WE)
        set(ARCHIVE_ZIP "${CMAKE_BINARY_DIR}/${FILENAME}.zip")
        list(APPEND ARCHIVE_ZIPS ${ARCHIVE_ZIP})

        file(DOWNLOAD
            ${URL}
            ${ARCHIVE_ZIP}
            SHOW_PROGRESS
            STATUS DOWNLOAD_STATUS
        )
        
        set(UNZIPPED_DIR "${DEST_DIR}/${FILENAME}")
        add_custom_command(
            OUTPUT ${UNZIPPED_DIR}
            COMMAND ${UNZIP_TOOL} ${ARCHIVE_ZIP} -d ${UNZIPPED_DIR} > /dev/null
            DEPENDS ${ARCHIVE_ZIP}
            COMMENT "Unzipping ${ARCHIVE_ZIP} into ${UNZIPPED_DIR}"
        )

        list(APPEND UNZIP_TARGETS ${UNZIPPED_DIR})
    endforeach()

    add_custom_target(DownloadAndUnzip ALL DEPENDS ${UNZIP_TARGETS})
endif()

if (BUILD_BENCH)
    add_subdirectory(bench)
endif()
