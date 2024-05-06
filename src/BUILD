cc_library(
    name = 'vkllama',
    srcs = [],
    hdrs = [],
    deps = ["//src/core:core", "//src/ops:ops"],
    copts = ['-std=c++17'],
    includes = ['./'],
    visibility = ['//visibility:public']
)

cc_library(
    name = 'models',
    srcs = [],
    hdrs = [],
    deps = ["//src:vkllama", "//src/models:llama2"],
    copts = ['-std=c++17'],
    includes = ['./'],
    visibility = ['//visibility:public']
)
