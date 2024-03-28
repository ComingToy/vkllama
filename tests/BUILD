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
    deps = ["//src:vkllama", "//shaders:vkllama_shaders", "@eigen3//:eigen"],
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
