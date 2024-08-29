cc_library(
    name = "llama2",
    srcs = [
        "tokenizer.cpp",
        "samplers.cpp",
    ],
    hdrs = [
        "llama2.h",
        "tokenizer.h",
        "samplers.h",
    ],
    deps = [
        "@gguf-tools//:gguf",
        "@sentencepiece//:sentencepiece"
    ],
    copts = ["-std=c++17"],
    visibility = ["//visibility:public"]
)
