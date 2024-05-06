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
	],
)

cc_binary(
    name = "test_rmsnorm",
    srcs = ["test_rmsnorm.cpp"],
    deps = ["//src:vkllama", "//src/shaders:vkllama_shaders", "@eigen3//:eigen"],
    copts = ["-std=c++17"],
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

cc_test(
	name = "test_multiheadattn",
	srcs = ["test_multiheadattn.cpp"],
    copts = ["-std=c++17"],
	deps = [
		":test_common",
	],
)

cc_test(
	name = "test_embedding",
	srcs = ["test_embedding.cpp"],
    copts = ["-std=c++17"],
	deps = [
		":test_common",
	],
)

cc_binary(
	name="llama2",
	srcs=["llama2.cpp"],
    copts = ["-std=c++17"],
    deps = ["//src/models:llama2", "//src:vkllama"],
)
