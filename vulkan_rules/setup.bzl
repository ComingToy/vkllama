load('//vulkan_rules:linux_lib.bzl', 'vulkan_linux_cc_lib')
load('//vulkan_rules:macos_lib.bzl', 'vulkan_macos_cc_lib')

def vulkan_setup():
    vulkan_linux_cc_lib(name='vulkan_linux')
    vulkan_macos_cc_lib(name='vulkan_macos')
    native.register_toolchains('//vulkan_rules:glsl_linux_toolchain')
    native.register_toolchains('//vulkan_rules:glsl_macos_toolchain')
