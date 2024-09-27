cc_library(
    name = "core",
    srcs = [
        "gpu_device.cpp",
        "pipeline.cpp",
        "tensor.cpp",
		"quants.cpp",
	],
    hdrs = [
        "command.h",
        "gpu_device.h",
        "pipeline.h",
        "tensor.h",
        "float.h",
        "shader_constants.h",
        "common.h",
        "quants.h",
	],
    deps = [
		"//vulkan_rules:vulkan_cc_library",
        "@abseil-cpp//absl/status:status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
		"@VulkanMemoryAllocator//:vma",
	],
    copts = select({
        "//:debug_build": ["-std=c++17", "-D__VKLLAMA_DEBUG__"],
        "//conditions:default": ["-std=c++17"]
    }),
    visibility = ["//visibility:public"]
)
