# vkllama

vkllama是一个使用vulkan作为后端开发的LLM推理库。
目前支持的模型
- llama2

目前支持的数据类型
- FP16
- Q8\_0

# 编译和运行
vkllama使用bazel进行编译。推荐使用[bazelisk](https://github.com/bazelbuild/bazelisk)安装对应的bazel版本。编译命令如下
```bash
bazel build //app:all
```
可以使用`app/llama_infer`来进行llama2的推理。
```bash
./bazel-bin/app/llama2_infer <path to gguf model> <path to tokenizer.model> <number of tokens> <prompt>
```
例如
```bash
./bazel-bin/app/llama2_infer Llama-2-7b-hf/Llama-2-7B-q8_0.gguf Llama-2-7b-hf/tokenizer.model 256 "Who is Linus Torvalds?"
```
输出
```
Who is Linus Torvalds?Linus Torvalds is a Finnish software engineer who is best known as the creator of the Linux kernel. He was born on
 December 28, 1969, in Helsinki, Finland.
Torvalds began working on the Linux kernel in 1991 while he was a student at the University of Helsinki. He released the first version of the kernel in 1994 and has since released numerous updates and improvements.
Torvalds is also the founder and former CEO of the Linux Foundation, a non-profit organization that promotes the use of Linux and open-source software.
Torvalds has received numerous awards and honors for his work on Linux, including the Millennium Technology Prize, the Turing Award, and
 the National Medal of Technology and Innovation.
Torvalds is known for his passionate and outspoken personality, and he has been a vocal advocate for open-source software and the free and open-source software movement.
Torvalds has also been involved in various other projects, including the Git version control system and the KDE desktop environment.
Torvalds is a strongeval speed: 15.4767 tokens/s
```
