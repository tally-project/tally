add_executable(run_transform run_transform.cpp)

target_include_directories(run_transform
    PUBLIC ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(run_transform PUBLIC
    tally
    PUBLIC Boost::serialization 
    PUBLIC Boost::regex
)