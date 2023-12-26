#pragma once
#include <map>
#include <optional>
#include <set>

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
          pushConstantRanges{std::move(o.pushConstantRanges)},
          descriptorSetLayouts{std::move(o.descriptorSetLayouts)},
          renderPass{o.renderPass},
          subpass{o.subpass} {
      o.reset();
    }
    ~PipelineBuilder() { reset(); }

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
      pushConstantRanges.clear();
      descriptorSetLayouts.clear();
      //
      renderPass = VK_NULL_HANDLE;
      subpass = 0;
    }

    /// default minimum setup
    void default_pipeline_configs();

    PipelineBuilder& setShader(vk::ShaderStageFlagBits stage, vk::ShaderModule shaderModule) {
      shaders[stage] = shaderModule;
      return *this;
    }
    PipelineBuilder& setShader(const ShaderModule& shaderModule);

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
    PipelineBuilder& setSubpass(u32 subpass) {
      this->subpass = subpass;
      return *this;
    }
    PipelineBuilder& setRenderPass(vk::RenderPass rp) {
      this->renderPass = rp;
      return *this;
    }

    /// @note provide alternatives for overwrite
    PipelineBuilder& setPushConstantRange(const vk::PushConstantRange& range) {
      this->pushConstantRanges = {range};
      return *this;
    }
    PipelineBuilder& setPushConstantRanges(const std::vector<vk::PushConstantRange>& ranges) {
      this->pushConstantRanges = ranges;
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

    PipelineBuilder& setBlendEnable(bool enable) {
      this->colorBlendAttachment.setBlendEnable(enable);
      return *this;
    }
    PipelineBuilder& setTopology(vk::PrimitiveTopology topology) {
      this->inputAssemblyInfo.setTopology(topology);
      return *this;
    }
    PipelineBuilder& setPolygonMode(vk::PolygonMode mode) {
      this->rasterizationInfo.setPolygonMode(mode);
      return *this;
    }
    PipelineBuilder& setCullMode(vk::CullModeFlagBits cm) {
      this->rasterizationInfo.setCullMode(cm);
      return *this;
    }
    PipelineBuilder& setRasterizationSamples(vk::SampleCountFlagBits sampleBits) {
      this->multisampleInfo.setRasterizationSamples(sampleBits);
      return *this;
    }

    PipelineBuilder& enableDynamicState(vk::DynamicState state) {
      dynamicStateEnables.insert(state);
      return *this;
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
    vk::PipelineInputAssemblyStateCreateInfo inputAssemblyInfo;
    vk::PipelineViewportStateCreateInfo viewportInfo;
    vk::PipelineRasterizationStateCreateInfo rasterizationInfo;
    vk::PipelineMultisampleStateCreateInfo multisampleInfo;
    //
    vk::PipelineDepthStencilStateCreateInfo depthStencilInfo;
    vk::PipelineColorBlendAttachmentState colorBlendAttachment;
    vk::PipelineColorBlendStateCreateInfo colorBlendInfo;
    std::set<vk::DynamicState> dynamicStateEnables;

    // resources (descriptors/ push constants)
    std::vector<vk::PushConstantRange> pushConstantRanges;
    std::map<u32, vk::DescriptorSetLayout> descriptorSetLayouts;  // managed outside

    /// render pass
    vk::RenderPass renderPass = VK_NULL_HANDLE;  // managed outside
    u32 subpass = 0;
  };

}  // namespace zs