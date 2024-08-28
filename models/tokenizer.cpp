#include "tokenizer.h"
#include <cstdio>

#if 0
	{LLM_KV_TOKENIZER_MODEL, "tokenizer.ggml.model"},
    {LLM_KV_TOKENIZER_PRE, "tokenizer.ggml.pre"},
    {LLM_KV_TOKENIZER_LIST, "tokenizer.ggml.tokens"},
    {LLM_KV_TOKENIZER_TOKEN_TYPE, "tokenizer.ggml.token_type"},
    {LLM_KV_TOKENIZER_TOKEN_TYPE_COUNT, "tokenizer.ggml.token_type_count"},
    {LLM_KV_TOKENIZER_SCORES, "tokenizer.ggml.scores"},
    {LLM_KV_TOKENIZER_MERGES, "tokenizer.ggml.merges"},
    {LLM_KV_TOKENIZER_BOS_ID, "tokenizer.ggml.bos_token_id"},
    {LLM_KV_TOKENIZER_EOS_ID, "tokenizer.ggml.eos_token_id"},
    {LLM_KV_TOKENIZER_UNK_ID, "tokenizer.ggml.unknown_token_id"},
    {LLM_KV_TOKENIZER_SEP_ID, "tokenizer.ggml.seperator_token_id"},
    {LLM_KV_TOKENIZER_PAD_ID, "tokenizer.ggml.padding_token_id"},
    {LLM_KV_TOKENIZER_CLS_ID, "tokenizer.ggml.cls_token_id"},
    {LLM_KV_TOKENIZER_MASK_ID, "tokenizer.ggml.mask_token_id"},
    {LLM_KV_TOKENIZER_ADD_BOS, "tokenizer.ggml.add_bos_token"},
    {LLM_KV_TOKENIZER_ADD_EOS, "tokenizer.ggml.add_eos_token"},
    {LLM_KV_TOKENIZER_ADD_PREFIX, "tokenizer.ggml.add_space_prefix"},
    {LLM_KV_TOKENIZER_REMOVE_EXTRA_WS,
     "tokenizer.ggml.remove_extra_whitespaces"},
    {LLM_KV_TOKENIZER_PRECOMPILED_CHARSMAP,
     "tokenizer.ggml.precompiled_charsmap"},
    {LLM_KV_TOKENIZER_HF_JSON, "tokenizer.huggingface.json"},
    {LLM_KV_TOKENIZER_RWKV, "tokenizer.rwkv.world"},
    {LLM_KV_TOKENIZER_PREFIX_ID, "tokenizer.ggml.prefix_token_id"},
    {LLM_KV_TOKENIZER_SUFFIX_ID, "tokenizer.ggml.suffix_token_id"},
    {LLM_KV_TOKENIZER_MIDDLE_ID, "tokenizer.ggml.middle_token_id"},
    {LLM_KV_TOKENIZER_EOT_ID, "tokenizer.ggml.eot_token_id"},
#endif

static sentencepiece::util::Status
load_tokenizer_pieces (std::map<std::string, gguf_key> const &gguf_kv,
                       sentencepiece::ModelProto &model)
{
  if (!gguf_kv.count ("tokenizer.ggml.tokens"))
    {
      return sentencepiece::util::Status (
          sentencepiece::util::StatusCode::kInvalidArgument,
          "miss key tokenizer.ggml.tokens");
    }

  if (!gguf_kv.count ("tokenizer.ggml.scores"))
    {
      return sentencepiece::util::Status (
          sentencepiece::util::StatusCode::kInvalidArgument,
          "miss key tokenizer.ggml.scores");
    }

  if (!gguf_kv.count ("tokenizer.ggml.token_type"))
    {
      return sentencepiece::util::Status (
          sentencepiece::util::StatusCode::kInvalidArgument,
          "miss key tokenizer.ggml.token_type");
    }

  auto tokens = gguf_kv.find ("tokenizer.ggml.tokens")->second;
  if (tokens.type != GGUF_VALUE_TYPE_ARRAY)
    {
      return sentencepiece::util::Status (
          sentencepiece::util::StatusCode::kInvalidArgument,
          "tokenizer.ggml.tokens is not array");
    }

  if (tokens.val->array.type != GGUF_VALUE_TYPE_STRING)
    {
      return sentencepiece::util::Status (
          sentencepiece::util::StatusCode::kInvalidArgument,
          "elements in tokenizer.ggml.tokens are not string type");
    }

  auto gguf_scores = gguf_kv.find ("tokenizer.ggml.scores")->second;
  if (gguf_scores.type != GGUF_VALUE_TYPE_ARRAY)
    {
      return sentencepiece::util::Status (
          sentencepiece::util::StatusCode::kInvalidArgument,
          "tokenizer.ggml.scores is not array");
    }

  if (gguf_scores.val->array.type != GGUF_VALUE_TYPE_FLOAT32)
    {
      return sentencepiece::util::Status (
          sentencepiece::util::StatusCode::kInvalidArgument,
          "elements in tokenizer.ggml.scores are not float32 type");
    }

  auto gguf_types = gguf_kv.find ("tokenizer.ggml.token_type")->second;
  if (gguf_types.type != GGUF_VALUE_TYPE_ARRAY)
    {
      return sentencepiece::util::Status (
          sentencepiece::util::StatusCode::kInvalidArgument,
          "tokenizer.ggml.token_type is not array");
    }

  if (gguf_types.val->array.type != GGUF_VALUE_TYPE_INT32)
    {
      return sentencepiece::util::Status (
          sentencepiece::util::StatusCode::kInvalidArgument,
          "elements in tokenizer.ggml.token_type are not int32 type");
    }

  std::vector<std::string> pieces;

  auto *val = reinterpret_cast<gguf_value *> (
      reinterpret_cast<uint8_t *> (tokens.val) + sizeof (tokens.val->array));

  for (size_t i = 0; i < tokens.val->array.len; ++i)
    {
      pieces.push_back (std::string (val->string.string, val->string.len));
      val = reinterpret_cast<gguf_value *> (reinterpret_cast<uint8_t *> (val)
                                            + 8 + val->string.len);
    }

  std::vector<float> scores;
  val = reinterpret_cast<gguf_value *> (
      reinterpret_cast<uint8_t *> (gguf_scores.val)
      + sizeof (gguf_scores.val->array));

  for (size_t i = 0; i < gguf_scores.val->array.len; ++i)
    {
      scores.push_back (val->float32);
      val = reinterpret_cast<gguf_value *> (reinterpret_cast<uint8_t *> (val)
                                            + sizeof (float));
    }

  std::vector<int> types;
  val = reinterpret_cast<gguf_value *> (
      reinterpret_cast<uint8_t *> (gguf_types.val)
      + sizeof (gguf_scores.val->array));

  for (size_t i = 0; i < gguf_types.val->array.len; ++i)
    {
      types.push_back ((int)val->int32);
      val = reinterpret_cast<gguf_value *> (reinterpret_cast<uint8_t *> (val)
                                            + sizeof (int32_t));
    }

  if (scores.size () != pieces.size () || scores.size () != types.size ())
    {
      return sentencepiece::util::Status (
          sentencepiece::util::StatusCode::kInvalidArgument,
          "len of tokenizer.ggml.scores != tokenizer.ggml.tokens || "
          "tokenizer.ggml.scores != tokenizer.ggml.token_type");
    }

  for (size_t i = 0; i < scores.size (); ++i)
    {
      auto p = model.add_pieces ();
      p->set_piece (pieces[i]);
      p->set_score (scores[i]);
      p->set_type ((::sentencepiece::ModelProto_SentencePiece_Type)types[i]);
    }

  return sentencepiece::util::Status ();
}

static sentencepiece::util::Status
load_trainer_spec (std::map<std::string, gguf_key> const &gguf_kv,
                   sentencepiece::ModelProto &model)
{
  std::string tokenizer_model = "llama";
  if (gguf_kv.count ("tokenizer.ggml.model") > 0)
    {
      auto tm = gguf_kv.find ("tokenizer.ggml.model")->second;
      if (tm.type == GGUF_VALUE_TYPE_STRING)
        tokenizer_model
            = std::string (tm.val->string.string, tm.val->string.len);
    }

  std::string tokenizer_pre = "default";
  if (gguf_kv.count ("tokenizer.ggml.pre") > 0)
    {
      auto pre = gguf_kv.find ("tokenizer.ggml.pre")->second;
      if (pre.type == GGUF_VALUE_TYPE_STRING)
        tokenizer_pre
            = std::string (pre.val->string.string, pre.val->string.len);
    }

  auto trainer_spec = model.mutable_trainer_spec ();
  trainer_spec->set_byte_fallback (true);
  if (tokenizer_model == "llama")
    {
      trainer_spec->set_model_type (sentencepiece::TrainerSpec_ModelType_BPE);
    }

  return sentencepiece::util::Status ();
}

sentencepiece::util::Status
load_tokenizer (sentencepiece::SentencePieceProcessor &sp,
                std::map<std::string, gguf_key> const &gguf_kv)
{

  sentencepiece::ModelProto model;
  if (auto s = load_tokenizer_pieces (gguf_kv, model); !s.ok ())
    {
      return s;
    }

  if (auto s = load_trainer_spec (gguf_kv, model); !s.ok ())
    {
      return s;
    }

  return sp.Load (model);
}
