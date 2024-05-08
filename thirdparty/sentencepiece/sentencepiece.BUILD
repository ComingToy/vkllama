load("@rules_cc//cc:defs.bzl", "cc_proto_library")
load("@rules_proto//proto:defs.bzl", "proto_library")

SPM_SRCS = ["src/*.cc", "third_party/absl/*/*.cc"]
SPM_HDRS = ["src/*.h", "third_party/absl/*/*.h", "third_party/darts_clone/*.h", "third_party/esaxx/*.hxx"]
SPM_PROTO = ["src/*.proto"]

cc_proto_library(
    name = "spm_cc_proto",
    deps = [":spm_proto"],
    visibility = ["//visibility:public"],
)

proto_library(
    name = "spm_proto",
    srcs = glob(SPM_PROTO),
)

genrule(
    name = 'config',
    srcs = ['config.h.in'],
    outs = ['config.h'],
    cmd = "cat $< | sed 's/@PROJECT_VERSION@/0.2.1/g' | sed 's/@PROJECT_NAME@/sentencepiece/g' > $@"
)

cc_library(
    name = "sentencepiece",
    srcs = glob(SPM_SRCS),
    hdrs = glob(SPM_HDRS) + ['config.h'],
    deps = [":config", ":spm_cc_proto"],
    copts = ["-std=c++17", "-D_USE_EXTERNAL_PROTOBUF"],
    includes = ["./src"],
    visibility = ['//visibility:public']
)


