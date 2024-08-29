#include "tokenizer.h"
#include <cstdio>

#if 0
It is not guaranteed to be standardized across models, and may change in the future. It is recommended that model authors use a more standardized tokenizer if possible.

tokenizer.ggml.model: string: The name of the tokenizer model.
llama: Llama style SentencePiece (tokens and scores extracted from HF tokenizer.model)
replit: Replit style SentencePiece (tokens and scores extracted from HF spiece.model)
gpt2: GPT-2 / GPT-NeoX style BPE (tokens extracted from HF tokenizer.json)
rwkv: RWKV tokenizer
tokenizer.ggml.tokens: array[string]: A list of tokens indexed by the token ID used by the model.
tokenizer.ggml.scores: array[float32]: If present, the score/probability of each token. If not present, all tokens are assumed to have equal probability. If present, it must have the same length and index as tokens.
tokenizer.ggml.token_type: array[int32]: The token type (1=normal, 2=unknown, 3=control, 4=user defined, 5=unused, 6=byte). If present, it must have the same length and index as tokens.
tokenizer.ggml.merges: array[string]: If present, the merges of the tokenizer. If not present, the tokens are assumed to be atomic.
tokenizer.ggml.added_tokens: array[string]: If present, tokens that were added after training.
Special tokens

tokenizer.ggml.bos_token_id: uint32: Beginning of sequence marker
tokenizer.ggml.eos_token_id: uint32: End of sequence marker
tokenizer.ggml.unknown_token_id: uint32: Unknown token
tokenizer.ggml.separator_token_id: uint32: Separator token
tokenizer.ggml.padding_token_id: uint32: Padding token
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

  auto trainer_spec = model.mutable_trainer_spec ();
  trainer_spec->set_byte_fallback (true);
  if (tokenizer_model == "llama")
    {
      trainer_spec->set_model_type (sentencepiece::TrainerSpec_ModelType_BPE);
    }

  if (gguf_kv.count ("tokenizer.ggml.bos_token_id") > 0)
    {
      auto bos = gguf_kv.find ("tokenizer.ggml.bos_token_id")->second;
      trainer_spec->set_bos_id ((int32_t)bos.val->uint32);
    }

  if (gguf_kv.count ("tokenizer.ggml.eos_token_id") > 0)
    {
      auto eos = gguf_kv.find ("tokenizer.ggml.eos_token_id")->second;
      trainer_spec->set_eos_id ((int32_t)eos.val->uint32);
    }

  if (gguf_kv.count ("tokenizer.ggml.unknown_token_id") > 0)
    {
      auto unk = gguf_kv.find ("tokenizer.ggml.unknown_token_id")->second;
      trainer_spec->set_unk_id ((int32_t)unk.val->uint32);
    }

  if (gguf_kv.count ("tokenizer.ggml.padding_token_id") > 0)
    {
      auto pad = gguf_kv.find ("tokenizer.ggml.padding_token_id")->second;
      trainer_spec->set_unk_id ((int32_t)pad.val->uint32);
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
