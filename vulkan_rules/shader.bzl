def _glsl_shader(ctx):
    toolchain = ctx.toolchains['//vulkan_rules:toolchain_type']
    shader = ctx.file.shader
    spv_name = shader.basename + '.spv'
    spv_file = ctx.actions.declare_file(spv_name, sibling=shader)
    
    args = ctx.actions.args()

    args.add('-o', spv_file.path)
    args.add(shader.path)

    ctx.actions.run(
        inputs = [shader],
        outputs = [spv_file],
        arguments = [args],
        executable = toolchain.glslc_executable,
        progress_message = 'compiling compute shader',
        mnemonic = 'GLSLC'
    )

    default_files = depset(direct = [spv_file])
    return [DefaultInfo(files=default_files)]

glsl_shader = rule(
    implementation = _glsl_shader,
    attrs = {'shader': attr.label(allow_single_file=['.comp'])},
    toolchains = ['//vulkan_rules:toolchain_type'],
    provides = [DefaultInfo]
)
