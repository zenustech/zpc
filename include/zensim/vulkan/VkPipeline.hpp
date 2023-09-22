#pragma once
#include <map>
#include <optional>

#include "zensim/vulkan/VkContext.hpp"
#include "zensim/vulkan/VkDescriptor.hpp"

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
          pipeline{VK_NULL_HANDLE},
          layout{VK_NULL_HANDLE} {}
    Pipeline(Pipeline&& o) noexcept
        : ctx{o.ctx},
          vertexShader{o.vertexShader},
          fragShader{o.fragShader},
          pipeline{o.pipeline},
          layout{o.layout} {
      o.vertexShader = VK_NULL_HANDLE;
      o.fragShader = VK_NULL_HANDLE;
      o.pipeline = VK_NULL_HANDLE;
      o.layout = VK_NULL_HANDLE;
    }
    ~Pipeline() {
      ctx.device.destroyPipeline(pipeline, nullptr, ctx.dispatcher);
      ctx.device.destroyPipelineLayout(layout, nullptr, ctx.dispatcher);
    }

    vk::Pipeline operator*() const { return pipeline; }
    operator vk::Pipeline() const { return pipeline; }
    operator vk::PipelineLayout() const { return layout; }

  protected:
    friend struct VulkanContext;
    friend struct PipelineBuilder;

    VulkanContext& ctx;
    vk::ShaderModule vertexShader, fragShader;
    /// @note manage the following constructs
    vk::Pipeline pipeline;
    vk::PipelineLayout layout;
  };

  struct PipelineBuilder {
    PipelineBuilder(VulkanContext& ctx) : ctx{ctx} { default_pipeline_configs(); }
    PipelineBuilder(PipelineBuilder&& o) noexcept
        : ctx{o.ctx},
          shaders{std::move(o.shaders)},
          inputAttributes{std::move(o.inputAttributes)},
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
          pushConstantRange{std::move(pushConstantRange)},
          descriptorSetLayouts{std::move(descriptorSetLayouts)},
          renderPass{o.renderPass},
          subpass{o.subpass} {
      o.reset();
    }
    ~PipelineBuilder() { reset(); }

    /// default minimum setup
    void default_pipeline_configs();

    PipelineBuilder& setShader(vk::ShaderStageFlagBits stage, vk::ShaderModule shaderModule) {
      shaders[stage] = shaderModule;
      return *this;
    }
    PipelineBuilder& setShader(const zs::ShaderModule& shaderModule);

    /// @note assume no padding and alignment involved
    /// @note if shaders are set through zs::ShaderModule and aos layout assumed, no need to
    /// explicitly configure input bindings here
    template <typename... ETs> PipelineBuilder& pushInputBinding(wrapt<ETs>...) {  // for aos layout
      constexpr int N = sizeof...(ETs);
      constexpr size_t szs[] = {sizeof(ETs)...};
      constexpr vk::Format fmts[N] = {deduce_attribute_format(wrapt<ETs>{})...};

      u32 binding = bindingDescriptions.size();
      u32 offset = 0;
      for (int i = 0; i < N; ++i) {
        attributeDescriptions.emplace_back(/*location*/ i,
                                           /*binding*/ binding, fmts[i],
                                           /*offset*/ offset);
        offset += szs[i];
      }
      bindingDescriptions.emplace_back(/*binding*/ binding,
                                       /*stride*/ (u32)offset, vk::VertexInputRate::eVertex);
      return *this;
    }

    /// @note if shaders are set through zs::ShaderModule, no need to explicitly configure
    /// descriptor set layouts anymore
    PipelineBuilder& addDescriptorSetLayout(vk::DescriptorSetLayout descrSetLayout, int no = -1) {
      if (no == -1) {
        descriptorSetLayouts[descriptorSetLayouts.size()] = descrSetLayout;
      } else
        descriptorSetLayouts[no] = descrSetLayout;
      return *this;
    }
    PipelineBuilder& setDescriptorSetLayouts(const std::map<u32, DescriptorSetLayout>& layouts,
                                             bool reset = false);
    PipelineBuilder& setRenderPass(vk::RenderPass rp) {
      this->renderPass = rp;
      return *this;
    }

    /// @note provide alternatives for overwrite
    PipelineBuilder& setPushConstantRange(const vk::PushConstantRange& range) {
      this->pushConstantRange = range;
      return *this;
    }
    PipelineBuilder& setBindingDescriptions(
        const std::vector<vk::VertexInputBindingDescription>& bindingDescriptions) {
      this->bindingDescriptions = bindingDescriptions;
      return *this;
    }
    PipelineBuilder& setAttributeDescriptions(
        const std::vector<vk::VertexInputAttributeDescription>& attributeDescriptions) {
      this->attributeDescriptions = attributeDescriptions;
      return *this;
    }

    //
    void reset() {
      shaders.clear();
      inputAttributes.clear();
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
      pushConstantRange.reset();
      descriptorSetLayouts.clear();
      //
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
    /// @note structure <binding, attributes (<location, <alignment bits, size, format, dims>>)>
    std::map<u32, AttributeDescriptor> inputAttributes;
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
    std::optional<vk::PushConstantRange> pushConstantRange;
    std::map<u32, vk::DescriptorSetLayout> descriptorSetLayouts;  // managed outside

    /// render pass
    vk::RenderPass renderPass = VK_NULL_HANDLE;  // managed outside
    u32 subpass = 0;
  };

}  // namespace zs