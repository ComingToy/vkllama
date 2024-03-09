cc_library(
    name = 'core',
    srcs = ['allocator.cpp', 'gpu_device.cpp', 'pipeline.cpp', 'tensor.cpp'],
    hdrs = ['allocator.h', 'command.h', 'gpu_device.h', 'pipeline.h', 'tensor.h'],
    deps = ["//vulkan_rules:vulkan_cc_library"],
    copts = ['-std=c++17'],
    visibility = ['//visibility:public']
)
