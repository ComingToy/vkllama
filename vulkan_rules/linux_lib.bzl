def _impl(ctx):
    "Vulkan linux repo"
    sdk_path = ctx.attr.path
    
    if sdk_path == '':
        sdk_path = ctx.os.environ.get('VULKAN_SDK', None)

    if sdk_path == '' or sdk_path == None:
        fail('Unable to locate vulkan sdk')

    ctx.symlink(sdk_path, 'vulkan_sdk_linux')

    glslc_path = ctx.path('vulkan_sdk_linux/bin/glslc')
    if not glslc_path.exists:
        fail('glslc not found.')
    
    file_content = """
cc_library(
    name = "vulkan_cc_library",
    srcs = ["vulkan_sdk_linux/lib/libvulkan.so"],  # replace lib_path
    hdrs = glob([
        "vulkan_sdk_linux/include/vulkan/*.h",
        "vulkan_sdk_linux/include/vulkan/*.hpp",
        ]),
    includes = ['vulkan_sdk_linux/include'],
    visibility = ["//visibility:public"]
)

filegroup(
    name = 'glslc',
    srcs = ['vulkan_sdk_linux/bin/glslc'],
    visibility = ['//visibility:public']
)
"""
    ctx.file('BUILD', file_content)


vulkan_linux_cc_lib = repository_rule(
    implementation = _impl,
    local = True,
    environ = ['VULKAN_SDK'],
    attrs = {
        'path': attr.string(mandatory = False)
    },
)
