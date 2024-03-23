cc_library(
    name = 'ops',
    srcs = ['op.cpp', 'mat_mul.cpp', 'rms_norm.cpp'],
    hdrs = ['op.h', 'mat_mul.h', 'rms_norm.h'],
    deps = ["//src/core:core", "//shaders:vkllama_shaders"],
    copts = ['-std=c++17'],
    visibility = ['//visibility:public']
)
