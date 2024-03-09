load(':toolchain.bzl', 'glsl_toolchain')
load('@rules_cc//cc:defs.bzl', 'cc_library')

config_setting(
    name = "macos",
    constraint_values = ["@platforms//os:macos"],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "linux",
    constraint_values = ["@platforms//os:linux"],
    visibility = ["//visibility:public"],
)

toolchain_type(
    name = "toolchain_type",
)

glsl_toolchain(
    name = 'linux_toolchain',
    glslc = '@vulkan_linux//:glslc',
)

toolchain(
    name = 'glsl_linux_toolchain',
    exec_compatible_with = [
        '@platforms//os:linux',
    ],
    toolchain = ':linux_toolchain',
    toolchain_type = ':toolchain_type'
)

glsl_toolchain(
    name = 'macos_toolchain',
    glslc = '@vulkan_macos//:glslc',
)

toolchain(
    name = 'glsl_macos_toolchain',
    exec_compatible_with = [
        '@platforms//os:macos',
    ],
    toolchain = ':macos_toolchain',
    toolchain_type = ':toolchain_type'
)

VULKAN_LIBS = select(
    {
        '//vulkan_rules:macos': ['@vulkan_macos//:vulkan_cc_library'],
        '//vulkan_rules:linux': ['@vulkan_linux//:vulkan_cc_library'],
    }
)

cc_library(
    name='vulkan_cc_library',
    visibility = ['//visibility:public'],
    deps = VULKAN_LIBS,
)
