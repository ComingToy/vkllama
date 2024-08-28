cc_library(
    name = "llama2",
    srcs = [
        "tokenizer.cpp"
    ],
    hdrs = [
        "llama2.h",
        "tokenizer.h",
    ],
    deps = [
        "@gguf-tools//:gguf",
        "@sentencepiece//:sentencepiece"
    ],
    copts = ["-std=c++17"],
    visibility = ["//visibility:public"]
)
