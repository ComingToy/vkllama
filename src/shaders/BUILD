load("//vulkan_rules:shader.bzl", "glsl_shader")
load("//vulkan_rules:shader.bzl", "cc_shader_library")

glsl_shader(
    name = "vkllama_comp_shaders",
    shaders = glob(["*.comp"]),
    hdrs = [
        "common.h",
        "header.h",
        "matmul_conf.h",
        "rms_norm_conf.h",
    ],
    extra_args = [
        "--target-env=vulkan1.2",
        "--target-spv=spv1.5",
    ],
    tool = "//tools:binary_to_array",
    visibility = ["//visibility:public"]
)

cc_shader_library(
    name = 'vkllama_shaders',
    deps = [':vkllama_comp_shaders'],
    visibility = ["//visibility:public"]
)
