#include "models/llama2.h"
#include "sentencepiece_processor.h"
#include <cstdio>
#include <unistd.h>

struct Params
{
  const char *model_file;
  const char *prompt_template;
  std::vector<const char *> anti_prompt;
  const char *sampler;
  struct
  {
    int topk;
    float p;
  } sampler_option;
};

#define _H(s) "\033[1m" #s "\033[0m"

static void
show_usage (int argc, char *const argv[])
{
  // clang-format off
  const char *fmt =
_H (NAME)"\n"
"    chat - chat with llama2 interactive \n\n"
_H (SYNOPSIS)"\n"
"    chat " _H(-m) " path " _H(-t) " path [" _H(-s) " {top_k|top_p}] [" _H(-k) " value] " "[" _H(-p) " value]" "\n"
"\n"
_H(DESCRIPTION)"\n"
"    the options are follow:\n"
"    " _H(-m) "\tpath to the gguf model file\n"
"    " _H(-t) "\tpath to the prompt template file\n"
"    " _H(-s) "\tsampler. top_k or top_p are supported. (default: top_k)\n"
"    " _H(-k) "\tthe k option of top_k sampler. (default: 40)\n"
"    " _H(-p) "\tthe p option of top_p sampler. (default: 0.75)\n"
;
  // clang-format on
  fprintf (stdout, fmt);
}

static int
parse_params_from_cmdline (int argc, char *const argv[], Params *params)
{
  int ch = -1;
  while ((ch = ::getopt (argc, argv, "m:t:a:s::k::p::")) != -1)
    {
      switch (ch)
        {
        case 'm':
          params->model_file = optarg;
        case 't':
          params->prompt_template = optarg;
          break;
        case 'a':
          params->anti_prompt.push_back (optarg);
          break;
        case 's':
          params->sampler = optarg;
          break;
        case 'k':
          params->sampler_option.topk = ::atoi (optarg);
          break;
        case 'p':
          params->sampler_option.p = ::atof (optarg);
          break;
        case '?':
        default:
          show_usage (argc, argv);
          return -1;
        }
    }

  return 0;
}

int
main (int argc, char *const argv[])
{
  int ret = 0;

  show_usage (argc, argv);
  Params params = { .model_file = nullptr,
                    .prompt_template = nullptr,
                    .anti_prompt = {},
                    .sampler = "topk",
                    .sampler_option = { .topk = 40, .p = 0.75 } };

  if ((ret = parse_params_from_cmdline (argc, argv, &params)) != 0)
    {
      return ret;
    }

  fprintf (
      stderr,
      "model_file: %s\nprompt_template: %s\nsampler: %s\ntopk: %d\np:%f\n",
      params.model_file, params.prompt_template, params.sampler,
      params.sampler_option.topk, params.sampler_option.p);
  for (size_t i = 0; i < params.anti_prompt.size (); ++i)
    {
      fprintf (stderr, "anti_prompt[%zu]: %s\n", i, params.anti_prompt[i]);
    }
}
