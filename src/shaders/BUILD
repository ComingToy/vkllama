load("//vulkan_rules:shader.bzl", "glsl_shader")
load("//vulkan_rules:shader.bzl", "cc_shader_library")

glsl_shader(
    name = "vkllama_comp_shaders",
    shaders = [
        "vec_mul_add.comp", 
        "rms_norm.comp",
        "element_wise.comp",
        "element_wise_fp16.comp",
        "element_wise_constant.comp",
        "element_wise_constant_fp16.comp",
        "matmul_broadcast0.comp",
        "matmul_broadcast1.comp",
        "matmul_broadcast2.comp",
        "matmul_broadcast0_fp16.comp",
        "matmul_broadcast1_fp16.comp",
        "matmul_broadcast2_fp16.comp",
        "rope.comp",
        "reduce_stage0.comp",
        "reduce_stage0_fp16.comp",
        "reduce_stage1.comp",
        "reduce_stage1_fp16.comp",
        "softmax_stage0.comp",
        "softmax_stage1.comp",
        "softmax_stage2.comp",
        "concat_axis2.comp",
        "concat_axis2_fp16.comp",
        "embedding.comp",
        "embedding_fp16.comp",
        "argmax_stage0.comp",
        "argmax_stage1.comp",
        "argmax_stage0_fp16.comp",
        "cast_fp32_to_fp16.comp",
        "cast_fp16_to_fp32.comp",
    ],
    tool = '//tools:binary_to_array',
    visibility = ["//visibility:public"]
)

cc_shader_library(
    name = 'vkllama_shaders',
    deps = [':vkllama_comp_shaders'],
    visibility = ["//visibility:public"]
)
