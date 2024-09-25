load("//vulkan_rules:shader.bzl", "glsl_shader")
load("//vulkan_rules:shader.bzl", "cc_shader_library")

glsl_shader(
    name = "vkllama_comp_shaders",
    shaders = glob(["*.comp"]),
    hdrs = [
        "common.h"
    ],
    extra_args = [
        "--target-spv=spv1.3",
    ],
    tool = "//tools:binary_to_array",
    visibility = ["//visibility:public"]
)

cc_shader_library(
    name = 'vkllama_shaders',
    deps = [':vkllama_comp_shaders'],
    visibility = ["//visibility:public"]
)
