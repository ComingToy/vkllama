load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def repo():
    git_repository(
        name = "abseil-cpp",
        remote = "https://github.com/abseil/abseil-cpp",
        tag = "20211102.0",
    )
