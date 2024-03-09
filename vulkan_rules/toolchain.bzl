def _glsl_toolchain_impl(ctx):
    glslc = ctx.executable.glslc

    toolchain_info = platform_common.ToolchainInfo(glslc=ctx.attr.glslc, glslc_executable=glslc)
    return [toolchain_info]


glsl_toolchain = rule(
    implementation = _glsl_toolchain_impl,
    attrs = {
        'glslc': attr.label(executable = True, cfg = 'exec', allow_single_file = True, mandatory = True)
    }
)
