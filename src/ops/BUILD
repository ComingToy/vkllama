cc_library(
    name = 'ops',
    srcs = [
        'op.cpp',
        'mat_mul.cpp',
        'rms_norm.cpp',
        'feed_forward.cpp',
        'rope.cpp',
		'elementwise.cpp',
    ],
    hdrs = [
        'op.h',
        'mat_mul.h',
        'rms_norm.h',
        'feed_forward.h',
        'rope.h',
		'elementwise.h',
		'reduce.h',
    ],
    deps = ["//src/core:core", "//shaders:vkllama_shaders"],
    copts = ['-std=c++17'],
    visibility = ['//visibility:public']
)
