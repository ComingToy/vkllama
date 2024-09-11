cc_library(
	name = "test_common",
	srcs = [],
	hdrs = ["test_common.h"],
    copts = ["-std=c++17"],
	deps = [
		"@eigen3//:eigen",
		"@gtest//:gtest",
		"@gtest//:gtest_main",
		"@//src:vkllama",
		"@abseil-cpp//absl/types:optional",
	],
)

cc_test(
	name = "test_rmsnorm",
	srcs = ["test_rmsnorm.cpp"],
    copts = ["-std=c++17"],
	deps = [
		":test_common",
	],
)

cc_test(
	name = "test_matmul",
	srcs = ["test_matmul.cpp"],
    copts = ["-std=c++17"],
	deps = [
		":test_common",
	],
)

cc_test(
	name = "test_feedforward",
	srcs = ["test_feedforward.cpp"],
    copts = ["-std=c++17"],
	deps = [
		":test_common",
	],
)

cc_test(
	name = "test_rope",
	srcs = ["test_rope.cpp"],
    copts = ["-std=c++17"],
	deps = [
		":test_common",
	],
)

cc_test(
	name = "test_elementwise",
	srcs = ["test_elementwise.cpp"],
    copts = ["-std=c++17"],
	deps = [
		":test_common",
	],
)

cc_test(
	name = "test_reduce",
	srcs = ["test_reduce.cpp"],
    copts = ["-std=c++17"],
	deps = [
		":test_common",
	],
)

cc_test(
	name = "test_softmax",
	srcs = ["test_softmax.cpp"],
    copts = ["-std=c++17"],
	deps = [
		":test_common",
	],
)

cc_test(
	name = "test_concat",
	srcs = ["test_concat.cpp"],
    copts = ["-std=c++17"],
	deps = [
		":test_common",
	],
)

#cc_test(
#	name = "test_multiheadattn",
#	srcs = ["test_multiheadattn.cpp"],
#    copts = ["-std=c++17"],
#	deps = [
#		":test_common",
#	],
#)

cc_test(
	name = "test_embedding",
	srcs = ["test_embedding.cpp"],
    copts = ["-std=c++17"],
	deps = [
		":test_common",
	],
)

cc_test(
	name = "test_argop",
	srcs = ["test_argop.cpp"],
    copts = ["-std=c++17"],
	deps = [
		":test_common",
	],
)

cc_test(
	name = "test_update_kv_cache",
	srcs = ["test_update_kv_cache.cpp"],
    copts = ["-std=c++17"],
	deps = [
		":test_common",
	],
)

cc_test(
	name = "test_slice",
	srcs = ["test_slice.cpp"],
    copts = ["-std=c++17"],
	deps = [
		":test_common",
	],
)

cc_binary(
	name="llama2",
	srcs=["llama2.cpp"],
    copts = ["-std=c++17"],
    deps = ["//models:llama2", "//src:vkllama", "@sentencepiece//:sentencepiece", "@gguf-tools//:gguf"],
)

cc_test(
    name = "test_transpose",
    srcs = ["test_transpose.cpp"],
    copts = ["-std=c++17"],
	deps = [
		":test_common",
	],
)

