load("//vulkan_rules:shader.bzl", "glsl_shader")
load(":cc_shader_library.bzl", "cc_shader_library")

glsl_shader(
    name = "vec_mul_add_shader",
    shader = "vec_mul_add.comp",
    visibility = ["//visibility:public"]
)

glsl_shader(
    name = "matmul_shader",
    shader = "matmul.comp",
    visibility = ["//visibility:public"]
)

glsl_shader(
    name = "matmul_smem_shader",
    shader = "matmul_shared_mem.comp",
    visibility = ["//visibility:public"]
)

cc_shader_library(
    name = 'vkllama_shaders',
    deps = [':vec_mul_add_shader', ':matmul_shader', ':matmul_smem_shader'],
    tool = '//tools:binary_to_array',
    visibility = ["//visibility:public"]
)
