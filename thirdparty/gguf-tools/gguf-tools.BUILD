cc_library(
    name = "gguf",
    srcs = glob(["*.c"], exclude=["gguf-tools.c"]),
    hdrs = glob(["*.h"]),
    visibility = ['//visibility:public'],
)

cc_binary(
    name = "gguf_tools",
    srcs = ['gguf-tools.c'],
    deps = [':gguf']
)
