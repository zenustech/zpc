#include "zensim/vulkan/VkPipeline.hpp"

#include <vulkan/vulkan_core.h>

#include "zensim/vulkan/VkBuffer.hpp"
#include "zensim/vulkan/VkImage.hpp"
#include "zensim/vulkan/VkRenderPass.hpp"

namespace zs {

  PipelineBuilder& PipelineBuilder::setDescriptorSetLayouts(
      const std::map<u32, DescriptorSetLayout>& layouts, bool reset) {
    if (reset) descriptorSetLayouts.clear();
    for (const auto& layout : layouts) descriptorSetLayouts[layout.first] = layout.second;
    return *this;
  }

  void PipelineBuilder::default_pipeline_configs() {
    shaders.clear();
    bindingDescriptions.clear();
    attributeDescriptions.clear();

    viewportInfo = vk::PipelineViewportStateCreateInfo{}
                       .setViewportCount(1)
                       .setPViewports(nullptr)
                       .setScissorCount(1)
                       .setPScissors(nullptr);

    inputAssemblyInfo = vk::PipelineInputAssemblyStateCreateInfo{}
                            .setTopology(vk::PrimitiveTopology::eTriangleList)
                            .setPrimitiveRestartEnable(false);

    rasterizationInfo = vk::PipelineRasterizationStateCreateInfo{}
                            .setDepthClampEnable(false)
                            .setRasterizerDiscardEnable(false)
                            .setPolygonMode(vk::PolygonMode::eFill)
                            .setLineWidth(1.f)
                            .setCullMode(vk::CullModeFlagBits::eNone)
                            .setFrontFace(vk::FrontFace::eClockwise)
                            .setDepthBiasEnable(false)
                            // optional
                            .setDepthBiasConstantFactor(0.f)
                            .setDepthBiasClamp(0.f)
                            .setDepthBiasSlopeFactor(0.f);

    multisampleInfo = vk::PipelineMultisampleStateCreateInfo{}
                          .setSampleShadingEnable(false)
                          .setRasterizationSamples(vk::SampleCountFlagBits::e1)
                          // optional
                          .setMinSampleShading(1.f)
                          .setPSampleMask(nullptr)
                          .setAlphaToCoverageEnable(false)
                          .setAlphaToOneEnable(false);

    colorBlendAttachment
        = vk::PipelineColorBlendAttachmentState{}
              .setColorWriteMask(vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG
                                 | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA)
              .setBlendEnable(false)
              // optional
              .setSrcColorBlendFactor(vk::BlendFactor::eOne)
              .setDstColorBlendFactor(vk::BlendFactor::eZero)
              .setColorBlendOp(vk::BlendOp::eAdd)
              .setSrcAlphaBlendFactor(vk::BlendFactor::eOne)
              .setDstAlphaBlendFactor(vk::BlendFactor::eZero)
              .setAlphaBlendOp(vk::BlendOp::eAdd);

    colorBlendInfo = vk::PipelineColorBlendStateCreateInfo{}
                         .setLogicOpEnable(false)
                         .setAttachmentCount(1)
                         .setPAttachments(&colorBlendAttachment)
                         // optional
                         .setLogicOp(vk::LogicOp::eCopy)
                         .setBlendConstants({0.f, 0.f, 0.f, 0.f});

    depthStencilInfo = vk::PipelineDepthStencilStateCreateInfo{}
                           .setDepthTestEnable(true)
                           .setDepthWriteEnable(true)
                           .setDepthCompareOp(vk::CompareOp::eLess)
                           .setDepthBoundsTestEnable(false)
                           .setStencilTestEnable(false)
                           // optional
                           .setMinDepthBounds(0.f)
                           .setMaxDepthBounds(1.f)
                           .setFront({})
                           .setBack({});

    dynamicStateEnables = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};

    pushConstantRanges.clear();
    descriptorSetLayouts.clear();

    renderPass = VK_NULL_HANDLE;
    subpass = 0;
  }

  Pipeline PipelineBuilder::build() {
    Pipeline ret{ctx};

    if (shaders.size() < 2) throw std::runtime_error("shaders are not fully prepared yet.");
    if (renderPass == VK_NULL_HANDLE) throw std::runtime_error("renderpass not yet specified.");

    // pipeline layout
    u32 nSets = descriptorSetLayouts.size();
    std::vector<vk::DescriptorSetLayout> descrSetLayouts(nSets);
    for (const auto& layout : descriptorSetLayouts) {
      if (layout.first < nSets) descrSetLayouts[layout.first] = layout.second;
    }
    auto pipelineLayout
        = ctx.device.createPipelineLayout(vk::PipelineLayoutCreateInfo{}
                                              .setSetLayoutCount(descrSetLayouts.size())
                                              .setPSetLayouts(descrSetLayouts.data())
                                              .setPushConstantRangeCount(pushConstantRanges.size())
                                              .setPPushConstantRanges(pushConstantRanges.data()),
                                          nullptr, ctx.dispatcher);
    ret.layout = pipelineLayout;

    // shaders
    std::vector<vk::PipelineShaderStageCreateInfo> shaderStages;
    for (const auto& iter : shaders) {
      shaderStages.emplace_back(vk::PipelineShaderStageCreateInfo{}
                                    .setStage(iter.first)
                                    .setModule(iter.second)
                                    .setPName("main")
                                    .setPNext(nullptr)
                                    .setPSpecializationInfo(nullptr));
    }

    // vertex input bindings
    auto vertexInputInfo = vk::PipelineVertexInputStateCreateInfo{}
                               .setVertexAttributeDescriptionCount(attributeDescriptions.size())
                               .setPVertexAttributeDescriptions(attributeDescriptions.data())
                               .setVertexBindingDescriptionCount(bindingDescriptions.size())
                               .setPVertexBindingDescriptions(bindingDescriptions.data());

    // dynamic state
    vk::PipelineDynamicStateCreateInfo dynamicStateInfo{
        {}, (u32)dynamicStateEnables.size(), dynamicStateEnables.data()};

    // pipeline
    auto pipelineInfo = vk::GraphicsPipelineCreateInfo{{},
                                                       (u32)shaderStages.size(),
                                                       shaderStages.data(),
                                                       &vertexInputInfo,
                                                       &inputAssemblyInfo,
                                                       /*tessellation*/ nullptr,
                                                       &viewportInfo,
                                                       &rasterizationInfo,
                                                       &multisampleInfo,
                                                       &depthStencilInfo,
                                                       &colorBlendInfo,
                                                       &dynamicStateInfo,
                                                       pipelineLayout,
                                                       renderPass,
                                                       subpass,
                                                       /*basePipelineHandle*/ VK_NULL_HANDLE,
                                                       /*basePipelineIndex*/ -1};

    if (ctx.device.createGraphicsPipelines(VK_NULL_HANDLE, (u32)1, &pipelineInfo, nullptr,
                                           &ret.pipeline, ctx.dispatcher)
        != vk::Result::eSuccess)
      throw std::runtime_error("failed to create graphics pipeline");

    return ret;
  }

}  // namespace zs