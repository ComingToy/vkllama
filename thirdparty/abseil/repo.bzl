load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def repo():
    git_repository(
        name = "abseil-cpp",
        remote = "https://github.com/abseil/abseil-cpp",
        tag = "20240722.0",
    )
