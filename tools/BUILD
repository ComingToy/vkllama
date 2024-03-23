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
