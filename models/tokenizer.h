#ifndef __VKLLAMA_TOKENIZER_H__
#define __VKLLAMA_TOKENIZER_H__
// clang-format off
#include <stddef.h>
extern "C"{
#include "gguflib.h"
}
// clang-format on
#include "sentencepiece_model.pb.h"
#include "sentencepiece_processor.h"

sentencepiece::util::Status
load_tokenizer (sentencepiece::SentencePieceProcessor &sp,
                std::map<std::string, gguf_key> const &gguf_kv);
#endif
