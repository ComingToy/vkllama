cc_library(
    name = 'core',
    srcs = ['gpu_device.cpp', 'pipeline.cpp', 'tensor.cpp'],
    hdrs = ['command.h', 'gpu_device.h', 'pipeline.h', 'tensor.h'],
    deps = ["//vulkan_rules:vulkan_cc_library", "@VulkanMemoryAllocator//:vma"],
    copts = ['-std=c++17'],
    visibility = ['//visibility:public']
)
