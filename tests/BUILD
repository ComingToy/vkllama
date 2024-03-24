cc_binary(
    name = "test_vec_add",
    srcs = ["test_vec_add.cpp"],
    deps = ["//src:vkllama", "//shaders:vkllama_shaders", "@eigen3//:eigen"],
    copts = ["-std=c++17"],
)

cc_binary(
    name = "test_smem_matmul",
    srcs = ["test_smem_matmul.cpp"],
    deps = ["//src:vkllama", "//shaders:vkllama_shaders", "@eigen3//:eigen"],
    copts = ["-std=c++17"],
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
		"//src:vkllama",
		"//shaders:vkllama_shaders",
		"@eigen3//:eigen",
		"@gtest//:gtest",
		"@gtest//:gtest_main",
	],
)
