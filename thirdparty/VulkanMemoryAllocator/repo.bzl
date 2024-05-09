load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def repo():
    git_repository(
        name = "VulkanMemoryAllocator",
        remote = "https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git",
        branch = "master",
        build_file = "//thirdparty/VulkanMemoryAllocator:VulkanMemoryAllocator.BUILD"
    )
