cc_library(
    name = 'ops',
    srcs = [
        'op.cpp',
        'mat_mul.cpp',
        'rms_norm.cpp',
        'feed_forward.cpp',
        'rope.cpp',
		'elementwise.cpp',
        'softmax.cpp',
    ],
    hdrs = [
        'op.h',
        'mat_mul.h',
        'rms_norm.h',
        'feed_forward.h',
        'rope.h',
		'elementwise.h',
		'reduce.h',
        'softmax.h'
    ],
    deps = ["//src/core:core", "//src/shaders:vkllama_shaders"],
    copts = ['-std=c++17'],
    visibility = ['//visibility:public']
)
