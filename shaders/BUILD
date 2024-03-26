load("//vulkan_rules:shader.bzl", "glsl_shader")
load("//vulkan_rules:shader.bzl", "cc_shader_library")

glsl_shader(
    name = "vkllama_comp_shaders",
    shaders = [
        "vec_mul_add.comp", 
        "matmul.comp",
        "matmul_shared_mem.comp",
        "rms_norm.comp",
        "element_wise.comp",
        "matmul_broadcast0.comp"
    ],
    tool = '//tools:binary_to_array',
    visibility = ["//visibility:public"]
)

cc_shader_library(
    name = 'vkllama_shaders',
    deps = [':vkllama_comp_shaders'],
    visibility = ["//visibility:public"]
)
