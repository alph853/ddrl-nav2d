set(DDRL_WORKER_PROTO_DIR ${CMAKE_CURRENT_SOURCE_DIR}/proto)
set(DDRL_WORKER_PROTO_SRCS
    ${DDRL_WORKER_PROTO_DIR}/rollouts.pb.cc
    ${DDRL_WORKER_PROTO_DIR}/rollouts.grpc.pb.cc
    ${DDRL_WORKER_PROTO_DIR}/policy.pb.cc
    ${DDRL_WORKER_PROTO_DIR}/policy.grpc.pb.cc
    ${DDRL_WORKER_PROTO_DIR}/coordinator.pb.cc
    ${DDRL_WORKER_PROTO_DIR}/coordinator.grpc.pb.cc
)

add_library(ddrl_worker_comm STATIC ${DDRL_WORKER_PROTO_SRCS})

target_include_directories(ddrl_worker_comm
    PUBLIC
        ${DDRL_WORKER_PROTO_DIR}
)

target_link_libraries(ddrl_worker_comm
    PUBLIC
        protobuf::libprotobuf
        gRPC::grpc++
        gRPC::grpc
)

add_executable(ddrl_worker)

find_package(CLI11 CONFIG REQUIRED)

# ------------------------------------------------------------------------
# ONNX Runtime (CPU)
set(ONNXRUNTIME_ROOT "${CMAKE_SOURCE_DIR}/third_party/onnxruntime" CACHE PATH "Path to ONNX Runtime installation")
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h PATHS ${ONNXRUNTIME_ROOT}/include)
find_library(ONNXRUNTIME_LIB onnxruntime PATHS ${ONNXRUNTIME_ROOT}/lib ${ONNXRUNTIME_ROOT}/lib64)

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIB)
    message(FATAL_ERROR
        "ONNX Runtime not found. Run scripts/install.sh "
        "or configure with -DONNXRUNTIME_ROOT=/path/to/onnxruntime."
    )
endif()

target_sources(ddrl_worker
    PRIVATE
        src/main.cpp
        src/worker.cpp
        src/config.cpp
        src/observation_encoder.cpp
        src/rollout_publisher.cpp
        src/trajectory_buffer.cpp
        src/policy_model.cpp
        src/coordinator_client.cpp
        src/policy_subscriber.cpp
        src/rollout_stream_client.cpp
        src/rollout_recorder.cpp
    PUBLIC
        FILE_SET HEADERS
        BASE_DIRS include
        FILES
            include/worker/worker.hpp
            include/worker/config.hpp
            include/worker/sim_instance.hpp
            include/worker/observation_encoder.hpp
            include/worker/rollout_publisher.hpp
            include/worker/trajectory_buffer.hpp
            include/worker/coordinator_client.hpp
            include/worker/policy_subscriber.hpp
            include/worker/rollout_stream_client.hpp
            include/worker/rollout_recorder.hpp
            include/worker/policy_model.hpp
            include/worker/viz_instance.hpp
)

target_include_directories(ddrl_worker
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${DDRL_WORKER_PROTO_DIR}
        ${ONNXRUNTIME_INCLUDE_DIR}
)

target_link_libraries(ddrl_worker
    PRIVATE
        ddrl::core
        ddrl::sim
        ddrl::events
        ddrl::config
        ddrl::visualizer
        # ddrl::rpc
        ddrl::logging
        ddrl::threading
        yaml-cpp
        CLI11::CLI11
        ddrl_worker_comm
        ${ONNXRUNTIME_LIB}
)
