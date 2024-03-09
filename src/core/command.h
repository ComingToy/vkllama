#ifndef __VKLLAMA_COMMAND_H__
#define __VKLLAMA_COMMAND_H__

#include "allocator.h"
#include "gpu_device.h"
#include "pipeline.h"
#include "tensor.h"
#include <cstddef>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>
#include <vulkan/vulkan_core.h>

class Command
{
  public:
    Command(GPUDevice* dev)
      : dev_(dev)
      , allocator_(dev->allocator())
    {
    }

    ~Command()
    {
		defer_task_.clear();
        vkDestroyCommandPool(dev_->device(), commandPool_, nullptr);
        vkDestroyFence(dev_->device(), fence_, nullptr);
    }

    VkResult init()
    {
        auto queueFamily =
          dev_->require_queue(VK_QUEUE_TRANSFER_BIT | VK_QUEUE_COMPUTE_BIT);
        vkGetDeviceQueue(dev_->device(), queueFamily, 0, &queue_);
        VkFenceCreateInfo fenceCreaeInfo = {
            VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, nullptr, 0
        };
        auto ret =
          vkCreateFence(dev_->device(), &fenceCreaeInfo, nullptr, &fence_);
        if (ret != VK_SUCCESS) {
            return ret;
        }

        VkCommandPoolCreateInfo createInfo = {
            VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            nullptr,
            VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            queueFamily
        };

        vkCreateCommandPool(
          dev_->device(), &createInfo, nullptr, &commandPool_);

        VkCommandBufferAllocateInfo allocInfo = {
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            nullptr,
            commandPool_,
            VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            1
        };

        return vkAllocateCommandBuffers(
          dev_->device(), &allocInfo, &commandBuffer_);
    }

    VkResult begin() { return begin_(); }
    VkResult end() { return end_(); }
    VkResult submit_and_wait() { return submit_and_wait_(); }

    template<typename T>
    VkResult upload(T const* from, const size_t n, VkTensor& to)
    {
        if (n * sizeof(T) < 65536) {
            return upload_small_(from, n, to);
        } else {
            return upload_large_(from, n, to);
        }
    }

    template<typename T>
    VkResult download(VkTensor& from, T* to, const size_t n)
    {
        const size_t aligned_bytes = ((n * sizeof(T) + 63) / 64) * 64;
        const size_t bytes = n * sizeof(T);
        if (from.visable()) {
            VkBufferMemoryBarrier barrier = {
                VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                nullptr,
                from.access_flags(),
                VK_ACCESS_HOST_READ_BIT,
                VK_QUEUE_FAMILY_IGNORED,
                VK_QUEUE_FAMILY_IGNORED,
                from.data(),
                0,
                aligned_bytes
            };

            vkCmdPipelineBarrier(commandBuffer_,
                                 from.pipeline_stage(),
                                 VK_PIPELINE_STAGE_HOST_BIT,
                                 0,
                                 0,
                                 nullptr,
                                 1,
                                 &barrier,
                                 0,
                                 nullptr);
            defer_task_.push_back([this, &from, to, bytes]() {
                auto ret = from.invalid();
                if (ret != VK_SUCCESS) {
                    return ret;
                }
                ::memcpy(to, from.host(), bytes);
                return VK_SUCCESS;
            });
            return VK_SUCCESS;
        }

        VkTensor staging(
          from.channels(), from.height(), from.width(), dev_, true);

        auto ret = staging.create();
        if (ret != VK_SUCCESS) {
            return ret;
        }

        {
            VkBufferMemoryBarrier barrier = {
                VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                nullptr,
                from.access_flags(),
                VK_ACCESS_TRANSFER_READ_BIT,
                VK_QUEUE_FAMILY_IGNORED,
                VK_QUEUE_FAMILY_IGNORED,
                from.data(),
                0,
                aligned_bytes
            };

            vkCmdPipelineBarrier(commandBuffer_,
                                 from.pipeline_stage(),
                                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 0,
                                 0,
                                 nullptr,
                                 1,
                                 &barrier,
                                 0,
                                 nullptr);
        }

        VkBufferCopy region = { 0, 0, aligned_bytes };
        vkCmdCopyBuffer(
          commandBuffer_, from.data(), staging.data(), 1, &region);

        {
            VkBufferMemoryBarrier barrier = {
                VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                nullptr,
                VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_ACCESS_HOST_READ_BIT,
                VK_QUEUE_FAMILY_IGNORED,
                VK_QUEUE_FAMILY_IGNORED,
                staging.data(),
                0,
                aligned_bytes
            };

            vkCmdPipelineBarrier(commandBuffer_,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_PIPELINE_STAGE_HOST_BIT,
                                 0,
                                 0,
                                 nullptr,
                                 1,
                                 &barrier,
                                 0,
                                 nullptr);
        }

        // defer to sumbit and wait then do read from this staging buffer
        defer_task_.push_back([this, staging, to, n, aligned_bytes]() mutable {
            auto ret = staging.flush();
            if (ret != VK_SUCCESS)
                return ret;
            ::memcpy(
              reinterpret_cast<void*>(to), staging.host(), sizeof(T) * n);
            return VK_SUCCESS;
        });
        return VK_SUCCESS;
    }

    VkResult record_pipeline(
      Pipeline& pipeline,
      std::vector<VkTensor> bindings,
      std::vector<Pipeline::ConstantType> const& constants)
    {
        auto& layout = pipeline.vklayout();
        auto& descriptset = pipeline.vkdescriptorset();

        for (auto& tensor : bindings) {
            VkBufferMemoryBarrier barrier = {
                VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                nullptr,
                tensor.access_flags(),
                VK_ACCESS_SHADER_READ_BIT,
                VK_QUEUE_FAMILY_IGNORED,
                VK_QUEUE_FAMILY_IGNORED,
                tensor.data(),
                0,
                tensor.bytes()
            };

            vkCmdPipelineBarrier(commandBuffer_,
                                 tensor.pipeline_stage(),
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 0,
                                 0,
                                 nullptr,
                                 1,
                                 &barrier,
                                 0,
                                 nullptr);
        }

        pipeline.update_bindings(bindings);

        vkCmdBindPipeline(
          commandBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.vkpileine());

        if (!constants.empty()) {
            vkCmdPushConstants(commandBuffer_,
                               layout,
                               VK_SHADER_STAGE_COMPUTE_BIT,
                               0,
                               sizeof(Pipeline::ConstantType) *
                                 constants.size(),
                               reinterpret_cast<const void*>(constants.data()));
        }

        vkCmdBindDescriptorSets(commandBuffer_,
                                VK_PIPELINE_BIND_POINT_COMPUTE,
                                layout,
                                0,
                                1,
                                &descriptset,
                                0,
                                nullptr);

        vkCmdResetQueryPool(commandBuffer_, pipeline.vkquerypool(), 0, 2);
        vkCmdWriteTimestamp(commandBuffer_,
                            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                            pipeline.vkquerypool(),
                            0);
        vkCmdDispatch(commandBuffer_,
                      pipeline.group_x(),
                      pipeline.group_y(),
                      pipeline.group_z());
        vkCmdWriteTimestamp(commandBuffer_,
                            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                            pipeline.vkquerypool(),
                            1);
        return VK_SUCCESS;
    }

  private:
    template<typename T>
    VkResult upload_small_(const T* from, const size_t n, VkTensor& to)
    {
        const size_t bytes = ((n * sizeof(T) + 63) / 64) * 64;
        vkCmdUpdateBuffer(commandBuffer_,
                          to.data(),
                          0,
                          bytes,
                          reinterpret_cast<const void*>(from));

        to.set_access_flags(VK_ACCESS_TRANSFER_WRITE_BIT);
        to.set_pipeline_stage(VK_PIPELINE_STAGE_TRANSFER_BIT);

        return VK_SUCCESS;
    }

    template<typename T>
    VkResult upload_large_(const T* from, size_t n, VkTensor& to)
    {
        const size_t aligned_bytes = (sizeof(T) * n + 63) / 64 * 64;
        const size_t bytes = n * sizeof(T);

        if (to.bytes() < bytes) {
            return VK_ERROR_OUT_OF_DEVICE_MEMORY;
        }

        if (to.visable()) {
            ::memcpy(to.host(), from, bytes);
            auto ret = to.invalid();
            if (ret != VK_SUCCESS) {
                return ret;
            }

            to.set_access_flags(VK_ACCESS_HOST_WRITE_BIT);
            to.set_pipeline_stage(VK_PIPELINE_STAGE_HOST_BIT);
            return VK_SUCCESS;
        }

        auto staging = std::make_shared<VkTensor>(
          to.channels(), to.height(), to.width(), dev_, true);

        auto ret = staging->create();
        if (ret != VK_SUCCESS)
            return ret;

        ::memcpy(staging->host(), reinterpret_cast<const void*>(from), bytes);

        ret = staging->invalid();
        if (ret != VK_SUCCESS) {
            return ret;
        }

        // put a barrier for host writing
        {
            VkBufferMemoryBarrier barrier = {
                VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                nullptr,
                VK_ACCESS_HOST_WRITE_BIT,
                VK_ACCESS_TRANSFER_READ_BIT,
                VK_QUEUE_FAMILY_IGNORED,
                VK_QUEUE_FAMILY_IGNORED,
                staging->data(),
                0,
                VK_WHOLE_SIZE
            };

            vkCmdPipelineBarrier(commandBuffer_,
                                 VK_PIPELINE_STAGE_HOST_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 0,
                                 0,
                                 nullptr,
                                 1,
                                 &barrier,
                                 0,
                                 nullptr);
        }

        VkBufferCopy region = { 0, 0, aligned_bytes };
        vkCmdCopyBuffer(commandBuffer_, staging->data(), to.data(), 1, &region);
        to.set_access_flags(VK_ACCESS_TRANSFER_WRITE_BIT);
        to.set_pipeline_stage(VK_PIPELINE_STAGE_TRANSFER_BIT);

        defer_task_.push_back([this, staging]() { return VK_SUCCESS; });
        return ret;
    }

    VkResult begin_()
    {
        VkCommandBufferBeginInfo info = {
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            nullptr,
            VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            nullptr
        };

        return vkBeginCommandBuffer(commandBuffer_, &info);
    }

    VkResult end_() { return vkEndCommandBuffer(commandBuffer_); }

    VkResult submit_and_wait_()
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

        auto ret = vkQueueSubmit(queue_, 1, &sumbitInfo, fence_);
        if (ret != VK_SUCCESS) {
            return ret;
        }

#if 1
        uint64_t timeout = 60ul * 1000000000ul; // 60s
        ret = vkWaitForFences(dev_->device(), 1, &fence_, true, timeout);
        if (ret != VK_SUCCESS) {
            return ret;
        }
        ret = vkResetFences(dev_->device(), 1, &fence_);
        if (ret != VK_SUCCESS) {
            return ret;
        }
#else
        vkQueueWaitIdle(queue_);
#endif

        VkResult defer_result = VK_SUCCESS;
        ret = VK_SUCCESS;

        for (auto& fn : defer_task_) {
            if ((defer_result = fn()) != VK_SUCCESS) {
                ret = defer_result;
            }
        }

        defer_task_.clear();
        return ret;
    }

    GPUDevice* dev_;
    VkQueue queue_;
    VkCommandBuffer commandBuffer_;
    VkFence fence_;
    VkCommandPool commandPool_;
    Allocator& allocator_;
    std::vector<std::function<VkResult(void)>> defer_task_;
};

class CommandScope
{
  public:
    CommandScope(GPUDevice* command)
      : command_(command)
    {
        command_.init();
        command_.begin();
    }

    template<typename T>
    VkResult upload(T const* from, const size_t n, VkTensor& to)
    {
        return command_.upload(from, n, to);
    }

    template<typename T>
    VkResult download(VkTensor& from, T* to, const size_t n)
    {
        return command_.download(from, to, n);
    }

    VkResult record_pipeline(
      Pipeline& pipeline,
      std::vector<VkTensor> bindings,
      std::vector<Pipeline::ConstantType> const& constants)
    {
        return command_.record_pipeline(pipeline, bindings, constants);
    }

    ~CommandScope()
    {
        command_.end();
        command_.submit_and_wait();
    }

  private:
    Command command_;
};
#endif
