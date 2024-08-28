#include "tokenizer.h"
sentencepiece::util::Status
load_tokenizer (sentencepiece::SentencePieceProcessor &sp,
                std::map<std::string, gguf_key> const &gguf_kv)
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

  sentencepiece::ModelProto model;
  for (size_t i = 0; i < scores.size (); ++i)
    {
      auto p = model.add_pieces ();
      p->set_piece (pieces[i]);
      p->set_score (scores[i]);
      p->set_type ((::sentencepiece::ModelProto_SentencePiece_Type)types[i]);
    }

  auto trainer_spec = model.mutable_trainer_spec ();
  trainer_spec->set_model_type (sentencepiece::TrainerSpec_ModelType_BPE);
  trainer_spec->set_byte_fallback (true);

  return sp.Load (model);
}
