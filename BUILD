load("@hedron_compile_commands//:refresh_compile_commands.bzl", "refresh_compile_commands")
exports_files([
    "configure",
    "configure.py",
    "ACKNOWLEDGEMENTS",
    "LICENSE",
])

refresh_compile_commands(
    name = "refresh_compile_commands",
    targets = {
        "//src:all": "",
        "//tests:all": "",
        "//app:all": "",
    },
)

config_setting (
    name = "linux",
    constraint_values = [
        "@platforms//os:linux"
    ],
    visibility = ["//visibility:public"]
)

config_setting (
    name = "windows",
    constraint_values = [
        "@platforms//os:windows"
    ],
    visibility = ["//visibility:public"]
)

config_setting(
    name = "debug_build",
    values = {
        "compilation_mode": 'dbg'
    },
)
