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
    };
    RenderPassBuilder(VulkanContext& ctx) noexcept
        : ctx{ctx}, _colorAttachments{}, _depthAttachment{} {}
    ~RenderPassBuilder() = default;

    RenderPassBuilder& addAttachment(vk::Format format = vk::Format::eR8G8B8A8Unorm,
                                     vk::AttachmentLoadOp op = vk::AttachmentLoadOp::eClear) {
      if (is_depth_format(format)) {
        _depthAttachment = AttachmentDesc{format, op};
      } else {
        _colorAttachments.push_back(AttachmentDesc{format, op});
      }
      return *this;
    }
    RenderPass build() const {
      RenderPass ret{ctx};
      std::vector<vk::AttachmentDescription> attachments;
      attachments.reserve(_colorAttachments.size() + (_depthAttachment ? 1 : 0));
      std::vector<vk::AttachmentReference> refs;
      refs.reserve(attachments.size());
      for (int i = 0; i != _colorAttachments.size(); ++i) {
        //
        refs.push_back(vk::AttachmentReference{(u32)attachments.size(),
                                               vk::ImageLayout::eColorAttachmentOptimal});
        //
        const auto& colorAttachmentDesc = _colorAttachments[i];
        attachments.push_back(vk::AttachmentDescription{}
                                  .setFormat(colorAttachmentDesc.format)
                                  .setSamples(vk::SampleCountFlagBits::e1)
                                  .setLoadOp(colorAttachmentDesc.loadOp)
                                  .setStoreOp(vk::AttachmentStoreOp::eStore)
                                  .setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
                                  .setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
                                  .setInitialLayout(vk::ImageLayout::eColorAttachmentOptimal)
                                  .setFinalLayout(vk::ImageLayout::eColorAttachmentOptimal));
      }

      auto subpass = vk::SubpassDescription{}
                         .setPipelineBindPoint(vk::PipelineBindPoint::eGraphics)
                         .setColorAttachmentCount((u32)attachments.size())
                         .setPColorAttachments(refs.data());

      if (_depthAttachment) {
        //
        refs.push_back(vk::AttachmentReference{(u32)attachments.size(),
                                               vk::ImageLayout::eDepthStencilAttachmentOptimal});
        //
        const auto& depthAttachmntDesc = *_depthAttachment;
        attachments.push_back(vk::AttachmentDescription{}
                                  .setFormat(depthAttachmntDesc.format)
                                  .setSamples(vk::SampleCountFlagBits::e1)
                                  .setLoadOp(depthAttachmntDesc.loadOp)
                                  .setStoreOp(vk::AttachmentStoreOp::eStore)
                                  .setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
                                  .setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
                                  .setInitialLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal)
                                  .setFinalLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal));
        //
        subpass.setPDepthStencilAttachment(&refs.back());
      }
      ret.renderpass = ctx.device.createRenderPass(vk::RenderPassCreateInfo{}
                                                       .setAttachmentCount(attachments.size())
                                                       .setPAttachments(attachments.data())
                                                       .setSubpassCount(1)
                                                       .setPSubpasses(&subpass)
                                                       .setDependencyCount(0)
                                                       .setPDependencies(nullptr),
                                                   nullptr, ctx.dispatcher);
      return ret;
    }

  private:
    VulkanContext& ctx;

    std::vector<AttachmentDesc> _colorAttachments;
    std::optional<AttachmentDesc> _depthAttachment;
  };

}  // namespace zs