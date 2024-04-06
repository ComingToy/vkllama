#include "mat_mul.h"
#include "src/shaders/vkllama_comp_shaders.h"
#include "src/core/command.h"
#include "src/core/gpu_device.h"
#include "src/core/pipeline.h"
#include "src/core/tensor.h"

MatMul::MatMul (GPUDevice *dev, Command *command, const int act,
                const int broadcast_type, const bool transpose_b)
    : Op (dev, command), broadcast_type_ (broadcast_type), act_ (act),
      transpose_b_ (transpose_b)
{
}

VkResult
MatMul::init ()
{
  Pipeline::ShaderInfo info = { 2, 3, 4, 16, 16, 1 };
  Pipeline::ConstantType act_type = { .i = act_ };
  Pipeline::ConstantType transpose_b
      = { .i = static_cast<int> (transpose_b_) };

  const uint8_t *pcode = nullptr;
  size_t code_size = 0;
  if (broadcast_type_ == 0)
    {
      pcode = __get_matmul_broadcast0_comp_spv_code ();
      code_size = __get_matmul_broadcast0_comp_spv_size ();
    }
  else if (broadcast_type_ == 1)
    {
      pcode = __get_matmul_broadcast1_comp_spv_code ();
      code_size = __get_matmul_broadcast1_comp_spv_size ();
    }
  else if (broadcast_type_ == 2)
    {
      pcode = __get_matmul_broadcast2_comp_spv_code ();
      code_size = __get_matmul_broadcast2_comp_spv_size ();
    }
  else
    {
      return VK_ERROR_UNKNOWN;
    }

  pipeline_.reset (
      new Pipeline (dev_, pcode, code_size, { act_type, transpose_b }, info));

  return pipeline_->init ();
}

uint64_t
MatMul::time ()
{
  return pipeline_->time ();
}

VkResult
MatMul::operator() (VkTensor a, VkTensor b, VkTensor &c)
{
  if ((broadcast_type_ == 0 && b.channels () != a.channels ())
      || (broadcast_type_ == 1 && b.channels () != 1))
    {
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
    }

  size_t out_h = a.height (), out_w = transpose_b_ ? b.height () : b.width ();
  c = VkTensor (std::max (a.channels (), b.channels ()), out_h, out_w, dev_,
                false);

  auto ret = c.create ();
  if (ret != VK_SUCCESS)
    {
      return ret;
    }

  int channels = std::max (a.channels (), b.channels ());
  Pipeline::ConstantType C = { .i = channels };
  Pipeline::ConstantType M = { .i = (int)a.height () };
  Pipeline::ConstantType N = { .i = (int)out_w };
  Pipeline::ConstantType K = { .i = (int)a.width () };

  uint32_t groupx = (N.i + 31) / 32, groupy = (M.i + 31) / 32, groupz = C.i;
  pipeline_->set_group (groupx, groupy, groupz);

  ret = command_->record_pipeline (*pipeline_, { a, b, c }, { C, M, N, K });
  if (ret != VK_SUCCESS)
    {
      return ret;
    }

  c.set_access_flags (VK_ACCESS_SHADER_WRITE_BIT);
  c.set_pipeline_stage (VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
  return VK_SUCCESS;
}
