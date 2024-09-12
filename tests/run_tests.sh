#!/bin/sh
bazel run //tests:test_argop
bazel run //tests:test_rmsnorm
bazel run //tests:test_matmul
bazel run //tests:test_feedforward
bazel run //tests:test_rope
bazel run //tests:test_elementwise
bazel run //tests:test_reduce
bazel run //tests:test_softmax
bazel run //tests:test_concat
bazel run //tests:test_embedding
bazel run //tests:test_update_kv_cache
bazel run //tests:test_slice
bazel run //tests:test_transpose
