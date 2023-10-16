#pragma once
#include "zensim/vulkan/VkContext.hpp"

namespace zs {

  /// @note https://www.khronos.org/blog/streamlining-render-passes
  struct RenderPass {
    RenderPass(VulkanContext& ctx) : ctx{ctx}, renderpass{VK_NULL_HANDLE} {}
    RenderPass(RenderPass&& o) noexcept : ctx{o.ctx}, renderpass{o.renderpass} {
      o.renderpass = VK_NULL_HANDLE;
    }
    ~RenderPass() { ctx.device.destroyRenderPass(renderpass, nullptr, ctx.dispatcher); }

    vk::RenderPass operator*() const { return renderpass; }
    operator vk::RenderPass() const { return renderpass; }

  protected:
    friend struct VulkanContext;
    friend struct Swapchain;
    friend struct RenderPassBuilder;

    VulkanContext& ctx;
    vk::RenderPass renderpass;
  };

  struct RenderPassBuilder {
    struct AttachmentDesc {
      vk::Format format;
      vk::AttachmentLoadOp loadOp;
      vk::ImageLayout layout;
    };
    RenderPassBuilder(VulkanContext& ctx) noexcept
        : ctx{ctx}, _colorAttachments{}, _depthAttachment{} {}
    ~RenderPassBuilder() = default;

    RenderPassBuilder& addAttachment(vk::Format format = vk::Format::eR8G8B8A8Unorm,
                                     vk::AttachmentLoadOp op = vk::AttachmentLoadOp::eClear,
                                     vk::ImageLayout layout
                                     = vk::ImageLayout::eColorAttachmentOptimal) {
      if (is_depth_format(format)) {
        _depthAttachment = AttachmentDesc{format, op, layout};
      } else {
        _colorAttachments.push_back(AttachmentDesc{format, op, layout});
      }
      return *this;
    }
    RenderPass build() const {
      RenderPass ret{ctx};
      const auto num = _colorAttachments.size() + (_depthAttachment ? 1 : 0);
      std::vector<vk::AttachmentDescription> attachments;
      attachments.reserve(num);
      std::vector<vk::AttachmentReference> refs;
      refs.reserve(num);
      for (int i = 0; i != _colorAttachments.size(); ++i) {
        const auto& colorAttachmentDesc = _colorAttachments[i];
        //
        refs.push_back(
            vk::AttachmentReference{(u32)attachments.size(), colorAttachmentDesc.layout});
        //
        attachments.push_back(vk::AttachmentDescription{}
                                  .setFormat(colorAttachmentDesc.format)
                                  .setSamples(vk::SampleCountFlagBits::e1)
                                  .setLoadOp(colorAttachmentDesc.loadOp)
                                  .setStoreOp(vk::AttachmentStoreOp::eStore)
                                  .setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
                                  .setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
                                  .setInitialLayout(colorAttachmentDesc.layout)
                                  .setFinalLayout(colorAttachmentDesc.layout));
      }

      auto subpass = vk::SubpassDescription{}
                         .setPipelineBindPoint(vk::PipelineBindPoint::eGraphics)
                         .setColorAttachmentCount((u32)attachments.size())
                         .setPColorAttachments(refs.data());

      if (_depthAttachment) {
        const auto& depthAttachmentDesc = *_depthAttachment;
        //
        refs.push_back(vk::AttachmentReference{
            (u32)attachments.size(),
            depthAttachmentDesc.layout});  // vk::ImageLayout::eDepthStencilAttachmentOptimal
        //
        attachments.push_back(vk::AttachmentDescription{}
                                  .setFormat(depthAttachmentDesc.format)
                                  .setSamples(vk::SampleCountFlagBits::e1)
                                  .setLoadOp(depthAttachmentDesc.loadOp)
                                  .setStoreOp(vk::AttachmentStoreOp::eStore)
                                  .setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
                                  .setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
                                  .setInitialLayout(depthAttachmentDesc.layout)
                                  .setFinalLayout(depthAttachmentDesc.layout));
        //
        subpass.setPDepthStencilAttachment(&refs.back());
      }
#if 0
      vk::AccessFlags accessFlag;
      if (_depthAttachment)
        accessFlag = vk::AccessFlagBits::eColorAttachmentWrite
                     | vk::AccessFlagBits::eDepthStencilAttachmentWrite;
      else
        accessFlag = vk::AccessFlagBits::eColorAttachmentWrite;
      auto dependency = vk::SubpassDependency{}
                            .setDstSubpass(0)
                            .setDstAccessMask(accessFlag)
                            .setDstStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput
                                             | vk::PipelineStageFlagBits::eEarlyFragmentTests)
                            .setSrcSubpass(VK_SUBPASS_EXTERNAL)
                            .setSrcAccessMask({})
                            .setSrcStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput
                                             | vk::PipelineStageFlagBits::eEarlyFragmentTests);
#endif

      ret.renderpass = ctx.device.createRenderPass(vk::RenderPassCreateInfo{}
                                                       .setAttachmentCount(attachments.size())
                                                       .setPAttachments(attachments.data())
                                                       .setSubpassCount(1)
                                                       .setPSubpasses(&subpass)
#if 1
                                                       .setDependencyCount(0)
                                                       .setPDependencies(nullptr)
#else
                                                       .setDependencyCount(1)
                                                       .setPDependencies(&dependency)
#endif
                                                       ,
                                                   nullptr, ctx.dispatcher);
      return ret;
    }

  private:
    VulkanContext& ctx;

    std::vector<AttachmentDesc> _colorAttachments;
    std::optional<AttachmentDesc> _depthAttachment;
  };

}  // namespace zs