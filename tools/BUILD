load("@rules_python//python:defs.bzl", "py_binary")
load("@pip_deps//:requirements.bzl", "requirement")

py_binary(
    name='binary_to_array',
    srcs=['binary_to_array.py'],
    visibility = ["//visibility:public"],
    deps = [
        requirement('jinja2')
    ]
)

py_binary(
    name='dump_all_variables',
    srcs=['dump_all_variables.py'],
    visibility = ["//visibility:public"],
    deps = [
        "//models/proto:llama2_model_py_proto",
        requirement('tensorflow')
    ]
)
