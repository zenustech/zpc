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
      vk::Format format{};
      vk::ImageLayout initialLayout{}, finalLayout{};
      vk::AttachmentLoadOp loadOp{vk::AttachmentLoadOp::eDontCare};
      vk::AttachmentStoreOp storeOp{vk::AttachmentStoreOp::eStore};
      vk::SampleCountFlagBits sampleBits{vk::SampleCountFlagBits::e1};
    };
    RenderPassBuilder(VulkanContext& ctx) noexcept
        : ctx{ctx}, _colorAttachments{}, _depthAttachment{}, _subpassCount{1} {}
    ~RenderPassBuilder() = default;

    RenderPassBuilder& addAttachment(const AttachmentDesc& desc) {
      // could check [desc] validity here
      if (is_depth_format(desc.format)) {
        _depthAttachment = desc;
      } else {
        _colorAttachments.push_back(desc);
      }
      return *this;
    }
    RenderPassBuilder& addAttachment(vk::Format format = vk::Format::eR8G8B8A8Unorm,
                                     vk::ImageLayout initialLayout = vk::ImageLayout::eUndefined,
                                     vk::ImageLayout finalLayout
                                     = vk::ImageLayout::eColorAttachmentOptimal,
                                     bool clear = true) {
      AttachmentDesc desc{format, initialLayout, finalLayout};
      if (clear)
        desc.loadOp = vk::AttachmentLoadOp::eClear;
      else {
        if (is_depth_format(format))
          desc.loadOp = vk::AttachmentLoadOp::eLoad;
        else
          desc.loadOp = initialLayout == vk::ImageLayout::eUndefined
                            ? vk::AttachmentLoadOp::eDontCare
                            : vk::AttachmentLoadOp::eLoad;
      }
      return addAttachment(desc);
    }
    RenderPassBuilder& addDepthAttachment(vk::Format format, bool clear) {
      AttachmentDesc desc{format, vk::ImageLayout::eDepthStencilAttachmentOptimal,
                          vk::ImageLayout::eDepthStencilAttachmentOptimal};
      desc.loadOp = clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad;
      return addAttachment(desc);
    }

    RenderPassBuilder& setNumPasses(u32 cnt) {
      _subpassCount = cnt;
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
        refs.push_back(vk::AttachmentReference{(u32)attachments.size(),
                                               vk::ImageLayout::eColorAttachmentOptimal});
        //
        attachments.push_back(vk::AttachmentDescription{}
                                  .setFormat(colorAttachmentDesc.format)
                                  .setSamples(colorAttachmentDesc.sampleBits)
                                  .setLoadOp(colorAttachmentDesc.loadOp)
                                  .setStoreOp(colorAttachmentDesc.storeOp)
                                  .setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
                                  .setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
                                  .setInitialLayout(colorAttachmentDesc.initialLayout)
                                  .setFinalLayout(colorAttachmentDesc.finalLayout));
      }
      if (_depthAttachment) {
        const auto& depthAttachmentDesc = *_depthAttachment;
        //
        refs.push_back(vk::AttachmentReference{
            (u32)attachments.size(),
            vk::ImageLayout::
                eDepthStencilAttachmentOptimal});  // vk::ImageLayout::eDepthStencilAttachmentOptimal
        //
        attachments.push_back(vk::AttachmentDescription{}
                                  .setFormat(depthAttachmentDesc.format)
                                  .setSamples(depthAttachmentDesc.sampleBits)
                                  .setLoadOp(depthAttachmentDesc.loadOp)
                                  .setStoreOp(depthAttachmentDesc.storeOp)
                                  .setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
                                  .setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
                                  .setInitialLayout(depthAttachmentDesc.initialLayout)
                                  .setFinalLayout(depthAttachmentDesc.finalLayout));
      }

      std::vector<vk::SubpassDescription> subpasses;
      std::vector<vk::SubpassDependency> subpassDependencies;

      vk::AccessFlags accessFlag;
      if (_depthAttachment)
        accessFlag = vk::AccessFlagBits::eColorAttachmentWrite
                     | vk::AccessFlagBits::eDepthStencilAttachmentWrite;
      else
        accessFlag = vk::AccessFlagBits::eColorAttachmentWrite;

      for (u32 i = 0; i < _subpassCount; i++) {
        auto subpass
            = vk::SubpassDescription{}
                  .setPipelineBindPoint(vk::PipelineBindPoint::eGraphics)
                  .setColorAttachmentCount((u32)attachments.size() - (_depthAttachment ? 1 : 0))
                  .setPColorAttachments(refs.data());
        if (_depthAttachment) subpass.setPDepthStencilAttachment(&refs.back());

        auto dependency
            = vk::SubpassDependency{}
                  .setSrcSubpass(i == 0 ? (VK_SUBPASS_EXTERNAL) : (i - 1))
                  .setDstSubpass(i)
                  .setSrcAccessMask({})
                  .setDstAccessMask(accessFlag)
                  .setSrcStageMask(
                      vk::PipelineStageFlagBits::
                          eColorAttachmentOutput /*vk::PipelineStageFlagBits::eEarlyFragmentTests*/)
                  .setDstStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);

        subpasses.push_back(subpass);
        subpassDependencies.push_back(dependency);
      }

      ret.renderpass
          = ctx.device.createRenderPass(vk::RenderPassCreateInfo{}
                                            .setAttachmentCount(attachments.size())
                                            .setPAttachments(attachments.data())
                                            .setSubpassCount(subpasses.size())
                                            .setPSubpasses(subpasses.data())
                                            .setDependencyCount(subpassDependencies.size())
                                            .setPDependencies(subpassDependencies.data()),
                                        nullptr, ctx.dispatcher);
      return ret;
    }

  private:
    VulkanContext& ctx;

    std::vector<AttachmentDesc> _colorAttachments;
    std::optional<AttachmentDesc> _depthAttachment;
    u32 _subpassCount;
  };

}  // namespace zs