load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def repo():
    git_repository(
        name = "gguf-tools",
        remote = "https://github.com/antirez/gguf-tools.git",
        branch = "main",
        build_file = "//thirdparty/gguf-tools:gguf-tools.BUILD"
    )
