#pragma once
#include "zensim/vulkan/Vulkan.hpp"

namespace zs {

  // (-1, -1) ---- x ----> (1, -1)
  //  |
  //  |
  //  y
  //  |
  //  |
  // (-1, 1)

  // ref: little vulkan engine
  struct PipelineConfig {
    std::vector<vk::VertexInputBindingDescription> bindingDescriptions{};
    std::vector<vk::VertexInputAttributeDescription> attributeDescriptions{};
    vk::PipelineViewportStateCreateInfo viewportInfo;
    vk::PipelineInputAssemblyStateCreateInfo inputAssemblyInfo;
    vk::PipelineRasterizationStateCreateInfo rasterizationInfo;
    vk::PipelineMultisampleStateCreateInfo multisampleInfo;
    vk::PipelineColorBlendAttachmentState colorBlendAttachment;
    vk::PipelineColorBlendStateCreateInfo colorBlendInfo;
    vk::PipelineDepthStencilStateCreateInfo depthStencilInfo;
    std::vector<vk::DynamicState> dynamicStateEnables;
    vk::PipelineDynamicStateCreateInfo dynamicStateInfo;
    vk::PipelineLayout pipelineLayout = nullptr;
    vk::RenderPass renderPass = nullptr;
    u32 subpass = 0;
  };

  struct Pipeline {
    Pipeline(VulkanContext& ctx)
        : ctx{ctx},
          vertexShader{VK_NULL_HANDLE},
          fragShader{VK_NULL_HANDLE},
          pipeline{VK_NULL_HANDLE} {}
    Pipeline(Pipeline&& o) noexcept
        : ctx{o.ctx}, vertexShader{o.vertexShader}, fragShader{o.fragShader}, pipeline{o.pipeline} {
      o.vertexShader = VK_NULL_HANDLE;
      o.fragShader = VK_NULL_HANDLE;
      o.pipeline = VK_NULL_HANDLE;
    }
    ~Pipeline() {
      ctx.device.destroyShaderModule(vertexShader, nullptr, ctx.dispatcher);
      ctx.device.destroyShaderModule(fragShader, nullptr, ctx.dispatcher);
      ctx.device.destroyPipeline(pipeline, nullptr, ctx.dispatcher);
    }

    vk::Pipeline operator*() const { return pipeline; }
    operator vk::Pipeline() const { return pipeline; }

  protected:
    friend struct VulkanContext;

    VulkanContext& ctx;
    vk::ShaderModule vertexShader, fragShader;
    vk::Pipeline pipeline;
  };

}  // namespace zs