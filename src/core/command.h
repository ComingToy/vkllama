#ifndef __VKLLAMA_COMMAND_H__
#define __VKLLAMA_COMMAND_H__

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "gpu_device.h"
#include "pipeline.h"
#include "tensor.h"
#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <vector>
#include <vulkan/vulkan_core.h>

namespace vkllama
{
class Command
{
public:
  Command (GPUDevice *dev) : dev_ (dev) {}

  ~Command ()
  {
    defer_task_.clear ();
    vkFreeCommandBuffers (dev_->device (), commandPool_, 1, &commandBuffer_);
    vkDestroyCommandPool (dev_->device (), commandPool_, nullptr);
    vkDestroyFence (dev_->device (), fence_, nullptr);
  }

  absl::Status
  init ()
  {
    auto queueFamily
        = dev_->require_queue (VK_QUEUE_TRANSFER_BIT | VK_QUEUE_COMPUTE_BIT);
    vkGetDeviceQueue (dev_->device (), queueFamily, 0, &queue_);
    VkFenceCreateInfo fenceCreaeInfo
        = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, nullptr, 0 };
    auto ret
        = vkCreateFence (dev_->device (), &fenceCreaeInfo, nullptr, &fence_);
    if (ret != VK_SUCCESS)
      {
        return absl::InternalError (absl::StrFormat (
            "Command: failed at create fence, VkResult = %d", int (ret)));
      }

    VkCommandPoolCreateInfo createInfo
        = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, nullptr,
            VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, queueFamily };

    vkCreateCommandPool (dev_->device (), &createInfo, nullptr, &commandPool_);

    VkCommandBufferAllocateInfo allocInfo
        = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr,
            commandPool_, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1 };

    ret = vkAllocateCommandBuffers (dev_->device (), &allocInfo,
                                    &commandBuffer_);
    if (ret != VK_SUCCESS)
      {
        return absl::InternalError (absl::StrFormat (
            "Command: failed at create fence, VkResult = %d", int (ret)));
      }

    return absl::OkStatus ();
  }

  absl::Status
  begin ()
  {
    return begin_ ();
  }
  absl::Status
  end ()
  {
    return end_ ();
  }
  absl::Status
  submit_and_wait ()
  {
    if (auto ret = submit (); !ret.ok ())
      {
        return ret;
      }
    return wait ();
  }

  template <typename T>
  absl::Status
  upload (T const *from, const size_t n, Tensor &to)
  {
    return upload_bytes (reinterpret_cast<const uint8_t *> (from),
                         n * sizeof (T), to);
  }

  absl::Status
  upload_bytes (uint8_t const *from, const size_t bytes, Tensor &to)
  {
    if (to.bytes () < bytes)
      {
        return absl::OutOfRangeError (absl::StrFormat (
            "to.size() = %zu but %zu bytes upload.", to.bytes (), bytes));
      }

    if (to.visable ())
      {
        ::memcpy (to.host (), from, bytes);
        auto ret = to.flush ();
        if (!ret.ok ())
          {
            return ret;
          }

        to.set_access_flags (VK_ACCESS_HOST_WRITE_BIT);
        to.set_pipeline_stage (VK_PIPELINE_STAGE_HOST_BIT);
        return absl::OkStatus ();
      }

    auto staging = std::make_shared<Tensor> (
        to.channels (), to.height (), to.width (), dev_, to.dtype (), true);

    auto ret = staging->create ();
    if (!ret.ok ())
      return ret;

    ::memcpy (staging->host (), reinterpret_cast<const void *> (from), bytes);

    ret = staging->flush ();
    if (!ret.ok ())
      {
        return ret;
      }

    // put a barrier for host writing
    {
      VkBufferMemoryBarrier barrier
          = { VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
              nullptr,
              VK_ACCESS_HOST_WRITE_BIT,
              VK_ACCESS_TRANSFER_READ_BIT,
              VK_QUEUE_FAMILY_IGNORED,
              VK_QUEUE_FAMILY_IGNORED,
              staging->data (),
              0,
              staging->bytes () };

      vkCmdPipelineBarrier (commandBuffer_, VK_PIPELINE_STAGE_HOST_BIT,
                            VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1,
                            &barrier, 0, nullptr);
    }

    VkBufferCopy region = { 0, 0, staging->bytes () };
    vkCmdCopyBuffer (commandBuffer_, staging->data (), to.data (), 1, &region);
    to.set_access_flags (VK_ACCESS_TRANSFER_WRITE_BIT);
    to.set_pipeline_stage (VK_PIPELINE_STAGE_TRANSFER_BIT);

    defer_task_.push_back ([staging] () { return absl::OkStatus (); });
    return ret;
  }

  template <typename T>
  absl::Status
  download (Tensor &from, T *to, const size_t n)
  {
    if (n < from.size ())
      {
        return absl::InternalError (
            absl::StrFormat ("download bytes larger than source tensor. "
                             "from.size() = %zu, to.size() = %zu",
                             from.size (), n));
      }

    if (from.visable ())
      {
        VkBufferMemoryBarrier barrier
            = { VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                nullptr,
                from.access_flags (),
                VK_ACCESS_HOST_READ_BIT,
                VK_QUEUE_FAMILY_IGNORED,
                VK_QUEUE_FAMILY_IGNORED,
                from.data (),
                0,
                from.bytes () };

        vkCmdPipelineBarrier (commandBuffer_, from.pipeline_stage (),
                              VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1,
                              &barrier, 0, nullptr);
        defer_task_.push_back ([&from, to] () {
          auto ret = from.invalid ();
          if (!ret.ok ())
            {
              return ret;
            }
          ::memcpy (to, from.host (), from.size () * sizeof (T));
          return absl::OkStatus ();
        });
        return absl::OkStatus ();
      }

    Tensor staging (1, 1, from.bytes (), dev_, INT8, true);

    auto ret = staging.create ();
    if (!ret.ok ())
      {
        return ret;
      }

    {
      VkBufferMemoryBarrier barrier
          = { VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
              nullptr,
              from.access_flags (),
              VK_ACCESS_TRANSFER_READ_BIT,
              VK_QUEUE_FAMILY_IGNORED,
              VK_QUEUE_FAMILY_IGNORED,
              from.data (),
              0,
              from.bytes () };

      vkCmdPipelineBarrier (commandBuffer_, from.pipeline_stage (),
                            VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1,
                            &barrier, 0, nullptr);
    }

    VkBufferCopy region = { 0, 0, from.bytes () };
    vkCmdCopyBuffer (commandBuffer_, from.data (), staging.data (), 1,
                     &region);

    {
      VkBufferMemoryBarrier barrier
          = { VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
              nullptr,
              VK_ACCESS_TRANSFER_WRITE_BIT,
              VK_ACCESS_HOST_READ_BIT,
              VK_QUEUE_FAMILY_IGNORED,
              VK_QUEUE_FAMILY_IGNORED,
              staging.data (),
              0,
              staging.bytes () };

      vkCmdPipelineBarrier (commandBuffer_, VK_PIPELINE_STAGE_TRANSFER_BIT,
                            VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1,
                            &barrier, 0, nullptr);
    }

    // defer to sumbit and wait then do read from this staging buffer
    defer_task_.push_back ([staging, to, n] () mutable {
      auto ret = staging.invalid ();
      if (!ret.ok ())
        return ret;
      ::memcpy (reinterpret_cast<void *> (to), staging.host (),
                sizeof (T) * n);
      return absl::OkStatus ();
    });
    return absl::OkStatus ();
  }

  absl::Status
  record_pipeline (Pipeline &pipeline, std::vector<Tensor> bindings,
                   std::vector<uint32_t> const &indices,
                   ShaderConstants const &constants)
  {
    auto &layout = pipeline.vklayout ();
    auto &descriptset = pipeline.vkdescriptorset ();

    for (auto &tensor : bindings)
      {
        if (tensor.access_flags () == 0 || tensor.pipeline_stage () == 0)
          {
            continue;
          }
        VkBufferMemoryBarrier barrier
            = { VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                nullptr,
                tensor.access_flags (),
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                VK_QUEUE_FAMILY_IGNORED,
                VK_QUEUE_FAMILY_IGNORED,
                tensor.data (),
                0,
                tensor.bytes () };

        vkCmdPipelineBarrier (commandBuffer_, tensor.pipeline_stage (),
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                              nullptr, 1, &barrier, 0, nullptr);
      }

    auto ret = pipeline.update_bindings (bindings, indices);
    if (!ret.ok ())
      {
        return ret;
      }

    vkCmdBindPipeline (commandBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE,
                       pipeline.vkpileine ());

    if (constants.elem_num () > 0)
      {
        vkCmdPushConstants (
            commandBuffer_, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
            constants.bytes (),
            reinterpret_cast<const void *> (constants.data ()));
      }

    vkCmdBindDescriptorSets (commandBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE,
                             layout, 0, 1, &descriptset, 0, nullptr);
    if (dev_->support_pipeline_statistics ())
      {
        vkCmdResetQueryPool (commandBuffer_, pipeline.vkquerypool (), 0, 2);
        vkCmdWriteTimestamp (commandBuffer_, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             pipeline.vkquerypool (), 0);
      }
    vkCmdDispatch (commandBuffer_, pipeline.group_x (), pipeline.group_y (),
                   pipeline.group_z ());
    if (dev_->support_pipeline_statistics ())
      {
        vkCmdWriteTimestamp (commandBuffer_,
                             VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                             pipeline.vkquerypool (), 1);
        defer_task_.push_back ([&pipeline] () {
          pipeline.query_exec_timestamp ();
          return absl::OkStatus ();
        });
      }

    return absl::OkStatus ();
  }

  absl::Status
  record_pipeline (Pipeline &pipeline, std::vector<Tensor> bindings,
                   ShaderConstants const &constants)
  {
    std::vector<uint32_t> indices;
    std::generate_n (std::back_inserter (indices), bindings.size (),
                     [i = 0u] () mutable { return i++; });

    return record_pipeline (pipeline, bindings, indices, constants);
  }

  absl::Status
  wait ()
  {
    uint64_t timeout = 60ul * 1000000000ul; // 60s
    auto ret = vkWaitForFences (dev_->device (), 1, &fence_, true, timeout);
    if (ret != VK_SUCCESS)
      {
        return absl::InternalError (
            absl::StrFormat ("failed at waiting fence: %d", int (ret)));
      }
    ret = vkResetFences (dev_->device (), 1, &fence_);
    if (ret != VK_SUCCESS)
      {
        return absl::InternalError (
            absl::StrFormat ("failed at reseting fence: %d", int (ret)));
      }

    absl::Status defer_result = absl::OkStatus ();

    for (auto &fn : defer_task_)
      {
        if (auto ret = fn (); !ret.ok ())
          {
            defer_result = ret;
          }
      }

    defer_task_.clear ();
    return defer_result;
  }

  absl::Status
  submit ()
  {
    VkSubmitInfo sumbitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO,
                                nullptr,
                                0,
                                nullptr,
                                nullptr,
                                1,
                                &commandBuffer_,
                                0,
                                nullptr };

    auto ret = vkQueueSubmit (queue_, 1, &sumbitInfo, fence_);
    if (ret != VK_SUCCESS)
      {
        return absl::InternalError (
            absl::StrFormat ("failed at submiting commands: %d\n", int (ret)));
      }

    return absl::OkStatus ();
  }

  void
  defer (std::function<absl::Status (void)> &&fn)
  {
    defer_task_.push_back (std::move (fn));
  }

private:
  absl::Status
  begin_ ()
  {
    VkCommandBufferBeginInfo info
        = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr,
            VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr };

    auto ret = vkBeginCommandBuffer (commandBuffer_, &info);
    if (ret != VK_SUCCESS)
      {
        return absl::InternalError (
            absl::StrFormat ("failed at commandbuffer begin: %d", int (ret)));
      }

    return absl::OkStatus ();
  }

  absl::Status
  end_ ()
  {
    auto ret = vkEndCommandBuffer (commandBuffer_);
    if (ret != VK_SUCCESS)
      {
        return absl::InternalError (
            absl::StrFormat ("failed at command end: %d", int (ret)));
      }

    return absl::OkStatus ();
  }

  GPUDevice *dev_;
  VkQueue queue_;
  VkCommandBuffer commandBuffer_;
  VkFence fence_;
  VkCommandPool commandPool_;
  std::vector<std::function<absl::Status (void)> > defer_task_;
};

class CommandScope
{
public:
  CommandScope (GPUDevice *command) : command_ (command)
  {
    (void)command_.init ();
    (void)command_.begin ();
  }

  ~CommandScope ()
  {
    (void)command_.end ();
    (void)command_.submit ();
  }

private:
  Command command_;
};
}

#endif
