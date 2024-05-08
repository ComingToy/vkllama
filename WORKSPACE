workspace(name = "vkllama")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

load('//vulkan_rules:setup.bzl', 'vulkan_setup')
vulkan_setup()

load('//thirdparty/eigen3:repo.bzl', eigen3_repo='repo')
eigen3_repo()

load('//thirdparty/gtest:repo.bzl', gtest_repo='repo')
gtest_repo()

SHA="84aec9e21cc56fbc7f1335035a71c850d1b9b5cc6ff497306f84cced9a769841"
VERSION="0.23.1"
http_archive(
    name = "rules_python",
    sha256 = SHA,
    strip_prefix = "rules_python-{}".format(VERSION),
    url = "https://github.com/bazelbuild/rules_python/releases/download/{}/rules_python-{}.tar.gz".format(VERSION,VERSION),
)
load("@rules_python//python:repositories.bzl", "py_repositories")
py_repositories()

load("@rules_python//python:pip.bzl", "pip_parse")
pip_parse(
   name = "pip_deps",
   requirements_lock = "//tools:requirements.txt",
)
load("@pip_deps//:requirements.bzl", "install_deps")
install_deps()

http_archive(
    name = "hedron_compile_commands",
    url = "https://github.com/hedronvision/bazel-compile-commands-extractor/archive/0e990032f3c5a866e72615cf67e5ce22186dcb97.tar.gz",
    strip_prefix = "bazel-compile-commands-extractor-0e990032f3c5a866e72615cf67e5ce22186dcb97",
)
load("@hedron_compile_commands//:workspace_setup.bzl", "hedron_compile_commands_setup")
hedron_compile_commands_setup()
load("@hedron_compile_commands//:workspace_setup_transitive.bzl", "hedron_compile_commands_setup_transitive")
hedron_compile_commands_setup_transitive()
load("@hedron_compile_commands//:workspace_setup_transitive_transitive.bzl", "hedron_compile_commands_setup_transitive_transitive")
hedron_compile_commands_setup_transitive_transitive()
load("@hedron_compile_commands//:workspace_setup_transitive_transitive_transitive.bzl", "hedron_compile_commands_setup_transitive_transitive_transitive")
hedron_compile_commands_setup_transitive_transitive_transitive()

http_archive(
    name = "rules_proto",
    sha256 = "80d3a4ec17354cccc898bfe32118edd934f851b03029d63ef3fc7c8663a7415c",
    strip_prefix = "rules_proto-5.3.0-21.5",
    urls = [
        "https://github.com/bazelbuild/rules_proto/archive/refs/tags/5.3.0-21.5.tar.gz",
    ],
)
load("@rules_proto//proto:repositories.bzl", "rules_proto_dependencies", "rules_proto_toolchains")
rules_proto_dependencies()
rules_proto_toolchains()

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
git_repository(
    name = "com_google_protobuf",
    remote = "https://github.com/protocolbuffers/protobuf",
    tag='v26.0'
)

load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")
protobuf_deps()

load("//thirdparty/sentencepiece:repo.bzl", sentencepiece_repo = "repo")
sentencepiece_repo()
