load("@bazel_tools//tools/cpp:toolchain_utils.bzl", "find_cpp_toolchain", "use_cpp_toolchain")
load("@rules_cc//cc:action_names.bzl", "C_COMPILE_ACTION_NAME")
load("@rules_cc//cc:action_names.bzl", "CPP_LINK_STATIC_LIBRARY_ACTION_NAME")

DISABLED_FEATURES = [
    # "module_maps",  # # copybara-comment-this-out-please
]

GlslInfo = provider(
    fields = [
        "srcs",
        "hdrs",
    ],
)

def _glsl_shader(ctx):
    toolchain = ctx.toolchains['//vulkan_rules:toolchain_type']
    output_spvs = []

    shaders = ctx.files.shaders
    for shader in shaders:
        spv_name = shader.basename + '.spv'
        spv_file = ctx.actions.declare_file(spv_name)
        print('genereate spv file: %s' % spv_name)
        
        args = ctx.actions.args()
        args.add('--target-spv=spv1.3')
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
        output_spvs.append(spv_file)

    output_header = ctx.actions.declare_file(ctx.label.name + '.h')
    output_cpp = ctx.actions.declare_file(ctx.label.name + '.cpp')
    print('genereate cpp files: %s' % output_cpp.path)
    args = [output_cpp.path, output_header.path] + [f.path for f in output_spvs]
    outputs = [output_cpp, output_header]
    ctx.actions.run(inputs=output_spvs, outputs=outputs, arguments=args, executable=ctx.executable.tool)

    srcs = depset(direct = [output_cpp])
    hdrs = depset(direct = [output_header])
    return [GlslInfo(srcs=srcs, hdrs=hdrs)]

def _cc_shader_library(ctx):
    output_file = ctx.actions.declare_file("lib" + ctx.label.name + ".a")
    cpp_files = []
    hdr_files = []

    for dep in ctx.attr.deps:
        infos = dep[GlslInfo]
        cpp_files += infos.srcs.to_list()
        hdr_files += infos.hdrs.to_list()

    obj_files = []
    cc_toolchain = find_cpp_toolchain(ctx)

    feature_configuration = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = cc_toolchain,
        requested_features = ctx.features,
        unsupported_features = DISABLED_FEATURES + ctx.disabled_features,
    )
    c_compiler_path = cc_common.get_tool_for_action(
        feature_configuration = feature_configuration,
        action_name = C_COMPILE_ACTION_NAME,
    )

    for cpp_file, hdr_file in zip(cpp_files, hdr_files):
        obj_file = ctx.actions.declare_file(cpp_file.basename + '.o')
        c_compile_variables = cc_common.create_compile_variables(
            feature_configuration = feature_configuration,
            cc_toolchain = cc_toolchain,
            user_compile_flags = ctx.fragments.cpp.copts + ctx.fragments.cpp.conlyopts,
            source_file = cpp_file.path,
            output_file = obj_file.path,
            quote_include_directories = depset([hdr_file.dirname]),
        )

        command_line = cc_common.get_memory_inefficient_command_line(
            feature_configuration = feature_configuration,
            action_name = C_COMPILE_ACTION_NAME,
            variables = c_compile_variables,
        )
        env = cc_common.get_environment_variables(
            feature_configuration = feature_configuration,
            action_name = C_COMPILE_ACTION_NAME,
            variables = c_compile_variables,
        )

        ctx.actions.run(
            executable = c_compiler_path,
            arguments = command_line,
            env = env,
            inputs = depset(
                [cpp_file, hdr_file],
                transitive = [cc_toolchain.all_files],
            ),
            outputs = [obj_file],
        )
        obj_files.append(obj_file)

    linker_input = cc_common.create_linker_input(
        owner = ctx.label,
        libraries = depset(direct = [
            cc_common.create_library_to_link(
                actions = ctx.actions,
                feature_configuration = feature_configuration,
                cc_toolchain = cc_toolchain,
                static_library = output_file,
                alwayslink = True,
            ),
        ]),
    )

    compilation_context = cc_common.create_compilation_context(headers=depset(hdr_files), quote_includes=depset([f.dirname for f in hdr_files]))
    linking_context = cc_common.create_linking_context(linker_inputs = depset(direct = [linker_input]))

    archiver_path = cc_common.get_tool_for_action(
        feature_configuration = feature_configuration,
        action_name = CPP_LINK_STATIC_LIBRARY_ACTION_NAME,
    )
    archiver_variables = cc_common.create_link_variables(
        feature_configuration = feature_configuration,
        cc_toolchain = cc_toolchain,
        output_file = output_file.path,
        is_using_linker = False,
    )
    command_line = cc_common.get_memory_inefficient_command_line(
        feature_configuration = feature_configuration,
        action_name = CPP_LINK_STATIC_LIBRARY_ACTION_NAME,
        variables = archiver_variables,
    )
    args = ctx.actions.args()
    args.add_all(command_line)
    for obj_file in obj_files:
        args.add(obj_file)

    env = cc_common.get_environment_variables(
        feature_configuration = feature_configuration,
        action_name = CPP_LINK_STATIC_LIBRARY_ACTION_NAME,
        variables = archiver_variables,
    )

    ctx.actions.run(
        executable = archiver_path,
        arguments = [args],
        env = env,
        inputs = depset(
            direct = [obj_file],
            transitive = [
                cc_toolchain.all_files,
            ],
        ),
        outputs = [output_file],
    )

    cc_info = cc_common.merge_cc_infos(cc_infos = [
        CcInfo(compilation_context = compilation_context, linking_context = linking_context),
    ])

    return [cc_info]

glsl_shader = rule(
    implementation = _glsl_shader,
    attrs = {
        'shaders': attr.label_list(allow_files=['.comp']),
        'tool': attr.label(executable=True, cfg='exec', allow_files=True),
    },
    toolchains = ['//vulkan_rules:toolchain_type'],
    provides = [GlslInfo]
)


cc_shader_library=rule(
    implementation=_cc_shader_library,
    attrs={
        'deps': attr.label_list(allow_files=True),
        "_cc_toolchain": attr.label(default = Label("@bazel_tools//tools/cpp:current_cc_toolchain")),
    },
    toolchains = use_cpp_toolchain(),
    fragments = ["cpp"],
)
