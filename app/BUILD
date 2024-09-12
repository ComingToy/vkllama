cc_binary(
	name="chat",
	srcs=["chat.cpp"],
    copts = ["-std=c++17"],
    deps = ["//models:llama2", "//src:vkllama"],
)

cc_binary(
	name="llama2_infer",
	srcs=["llama2_infer.cpp"],
    copts = ["-std=c++17"],
    deps = ["//models:llama2", "//src:vkllama"],
)
