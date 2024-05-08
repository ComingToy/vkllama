load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def repo():
    git_repository(
        name = "sentencepiece",
        remote = "https://github.com/google/sentencepiece.git",
        branch = "master",
        build_file = "//thirdparty/sentencepiece:sentencepiece.BUILD"
    )
