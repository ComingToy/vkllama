# vkllama

vkllama is an LLM inference library developed with vulkan as the backend.
Currently supported models
- llama2

Currently supported dtypes
- FP16
- Q8\_0

# Compile and run
we use bazel for compiling vkllama. It is recommended to use [bazelisk](https://github.com/bazelbuild/bazelisk) to install bazel. The compilation command is as follows
```bash
bazel build //app:all
```
You can use `app/llama_infer` inference llama2.
```bash 
./bazel-bin/app/llama2_infer <path to gguf model> <path to tokenizer.model> <number of tokens> <prompt> 
```
For example 
```bash
./bazel-bin/app/llama2_infer Llama-2-7b-hf/Llama-2-7B-q8_0.gguf Llama-2-7b-hf/tokenizer.model 256 "Who is Linus Torvalds?" 
``` 
output 
`Who is Linus Torvalds?Linus Torvalds is a Finnish software engineer who is best known as the creator of the Linux kernel. He was born on December 28, 1969, in Helsinki, Finland. Torvalds began working on the Linux kernel in 1991 while he was a student at the University of Helsinki. He released the first version of the kernel in 1994 and has since released numerous updates and improvements. Torvalds is also the founder and former CEO of the Linux Foundation, a non-profit organization that promotes the use of Linux and open-source software. Torvalds has received numerous awards and honors for his work on Linux, including the Millennium Technology Prize, the Turing Award, and the National Medal of Technology and Innovation. Torvalds is known for his passionate and outspoken personality, and he has been a vocal advocate for open-source software and the free and open-source software movement. Torvalds has also been involved in various other projects, including the Git version control system and the KDE desktop environment. Torvalds is a strong`
