#pragma once
#include <map>

#include "zensim/vulkan/VkContext.hpp"

namespace zs {

  // ref: little vulkan engine
  // (-1, -1) ---- x ----> (1, -1)
  //  |
  //  |
  //  y
  //  |
  //  |
  // (-1, 1)

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
    friend struct PipelineBuilder;

    VulkanContext& ctx;
    vk::ShaderModule vertexShader, fragShader;
    vk::Pipeline pipeline;
  };

  struct PipelineBuilder {
    PipelineBuilder(VulkanContext& ctx) : ctx{ctx} { default_pipeline_configs(); }
    PipelineBuilder(PipelineBuilder&& o) noexcept
        : ctx{o.ctx},
          shaders{std::move(o.shaders)},
          bindingDescriptions{std::move(o.bindingDescriptions)},
          attributeDescriptions{std::move(o.attributeDescriptions)},
          viewportInfo{o.viewportInfo},
          inputAssemblyInfo{o.inputAssemblyInfo},
          rasterizationInfo{o.rasterizationInfo},
          multisampleInfo{o.multisampleInfo},
          colorBlendAttachment{o.colorBlendAttachment},
          colorBlendInfo{o.colorBlendInfo},
          depthStencilInfo{o.depthStencilInfo},
          dynamicStateEnables{std::move(o.dynamicStateEnables)},
          pushConstantRanges{std::move(pushConstantRanges)},
          descriptorSetLayouts{std::move(descriptorSetLayouts)},
          pipelineLayout{o.pipelineLayout},
          renderPass{o.renderPass},
          subpass{o.subpass} {
      o.reset();
    }
    ~PipelineBuilder() { reset(); }

    /// default minimum setup
    void default_pipeline_configs();
    template <typename ET> PipelineBuilder& pushInputBinding(wrapt<ET>) {  // for soa layout
      bindingDescriptions.emplace_back(/*binding*/ bindingDescriptions.size(),
                                       /*stride*/ (u32)sizeof(ET), vk::VertexInputRate::eVertex);
      attributeDescriptions.emplace_back(/*location*/ 0, /*binding*/ bindingDescriptions.size() - 1,
                                         deduce_attribute_format(wrapt<ET>{}), /*offset*/ (u32)0);
      return *this;
    }
    PipelineBuilder& setShader(vk::ShaderStageFlagBits stage, vk::ShaderModule shaderModule) {
      shaders[stage] = shaderModule;
      return *this;
    }
    PipelineBuilder& addDescriptorSetLayout(vk::DescriptorSetLayout descrSetLayout) {
      descriptorSetLayouts.push_back(descrSetLayout);
      return *this;
    }
    PipelineBuilder& setRenderPass(vk::RenderPass rp) {
      this->renderPass = rp;
      return *this;
    }
    //
    void reset() {
      shaders.clear();
      bindingDescriptions.clear();
      attributeDescriptions.clear();
      viewportInfo = vk::PipelineViewportStateCreateInfo{};
      inputAssemblyInfo = vk::PipelineInputAssemblyStateCreateInfo{};
      rasterizationInfo = vk::PipelineRasterizationStateCreateInfo{};
      multisampleInfo = vk::PipelineMultisampleStateCreateInfo{};
      colorBlendAttachment = vk::PipelineColorBlendAttachmentState{};
      colorBlendInfo = vk::PipelineColorBlendStateCreateInfo{};
      depthStencilInfo = vk::PipelineDepthStencilStateCreateInfo{};
      dynamicStateEnables.clear();
      pushConstantRanges.clear();
      descriptorSetLayouts.clear();
      //
      ctx.device.destroyPipelineLayout(pipelineLayout, nullptr, ctx.dispatcher);
      pipelineLayout = VK_NULL_HANDLE;
      renderPass = VK_NULL_HANDLE;
      subpass = 0;
    }

    Pipeline build();

  protected:
    friend struct VulkanContext;

    VulkanContext& ctx;

    ///
    std::map<vk::ShaderStageFlagBits, vk::ShaderModule> shaders;  // managed outside
    /// fixed function states
    // vertex input state
    std::vector<vk::VertexInputBindingDescription> bindingDescriptions{};
    std::vector<vk::VertexInputAttributeDescription> attributeDescriptions{};
    vk::PipelineViewportStateCreateInfo viewportInfo;
    vk::PipelineInputAssemblyStateCreateInfo inputAssemblyInfo;
    vk::PipelineRasterizationStateCreateInfo rasterizationInfo;
    vk::PipelineMultisampleStateCreateInfo multisampleInfo;
    //
    vk::PipelineColorBlendAttachmentState colorBlendAttachment;
    vk::PipelineColorBlendStateCreateInfo colorBlendInfo;
    vk::PipelineDepthStencilStateCreateInfo depthStencilInfo;
    std::vector<vk::DynamicState> dynamicStateEnables;

    // resources (descriptors/ push constants)
    std::vector<vk::PushConstantRange> pushConstantRanges;
    std::vector<vk::DescriptorSetLayout> descriptorSetLayouts;  // managed outside

    vk::PipelineLayout pipelineLayout = VK_NULL_HANDLE;  // manage this
    /// render pass
    vk::RenderPass renderPass = VK_NULL_HANDLE;  // managed outside
    u32 subpass = 0;
  };

}  // namespace zs