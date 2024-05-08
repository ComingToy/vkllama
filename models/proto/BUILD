load("@rules_cc//cc:defs.bzl", "cc_proto_library")
load("@rules_proto//proto:defs.bzl", "proto_library")
load("@com_google_protobuf//:protobuf.bzl", "py_proto_library")

py_proto_library(
    name = "llama2_model_py_proto",
    srcs = ["llama2_model.proto"],
    visibility = ["//visibility:public"]
)

cc_proto_library(
    name = "llama2_model_cc_proto",
    deps = [":llama2_model_proto"],
    visibility = ["//visibility:public"]
)

proto_library(
    name = "llama2_model_proto",
    srcs = ["llama2_model.proto"],
)

