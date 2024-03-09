
load("@bazel_tools//tools/cpp:toolchain_utils.bzl", "find_cpp_toolchain", "use_cpp_toolchain")
load("@rules_cc//cc:action_names.bzl", "C_COMPILE_ACTION_NAME")
load("@rules_cc//cc:action_names.bzl", "CPP_LINK_STATIC_LIBRARY_ACTION_NAME")

DISABLED_FEATURES = [
    # "module_maps",  # # copybara-comment-this-out-please
]

def _cc_shader_library(ctx):
    output_cpp = ctx.actions.declare_file(ctx.label.name + '.cpp')
    output_header = ctx.actions.declare_file(ctx.label.name + '.h')
    obj_file = ctx.actions.declare_file(ctx.label.name + ".o")
    output_file = ctx.actions.declare_file("lib" + ctx.label.name + ".a")

    args = [output_cpp.path, output_header.path] + [f.path for f in ctx.files.deps]
    ctx.actions.run(inputs=ctx.files.deps, outputs=[output_cpp, output_header], arguments=args, executable=ctx.executable.tool)

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
    c_compile_variables = cc_common.create_compile_variables(
        feature_configuration = feature_configuration,
        cc_toolchain = cc_toolchain,
        user_compile_flags = ctx.fragments.cpp.copts + ctx.fragments.cpp.conlyopts,
        source_file = output_cpp.path,
        output_file = obj_file.path,
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
            [output_cpp],
            transitive = [cc_toolchain.all_files],
        ),
        outputs = [obj_file],
    )

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

    compilation_context = cc_common.create_compilation_context(headers=depset([output_header]), quote_includes=depset([output_header.dirname]))
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

    default_files = depset([output_file, output_header, output_cpp])
    return [DefaultInfo(files=default_files), cc_info]

cc_shader_library=rule(
    implementation=_cc_shader_library,
    attrs={
        'deps': attr.label_list(allow_files=True),
        'tool': attr.label(executable=True, cfg='exec', allow_files=True),
        "_cc_toolchain": attr.label(default = Label("@bazel_tools//tools/cpp:current_cc_toolchain")),
    },
    toolchains = use_cpp_toolchain(),
    fragments = ["cpp"],
)
