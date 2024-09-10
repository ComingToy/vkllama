load("@rules_python//python:defs.bzl", "py_binary")
load("@python_libs//:requirements.bzl", "requirement")
load("@python_macos_libs//:requirements.bzl", macos_requirement = "requirement")

py_binary(
    name='binary_to_array',
    srcs=['binary_to_array.py'],
    visibility = ["//visibility:public"],
    deps = select({
        '@platforms//os:macos': [macos_requirement('jinja2')],
        '@platforms//os:linux': [requirement('jinja2')]
    })
)
