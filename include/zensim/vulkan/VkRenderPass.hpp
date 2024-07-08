#pragma once
#include "VkUtils.hpp"
#include "zensim/ZpcTuple.hpp"
#include "zensim/vulkan/VkContext.hpp"

namespace zs {

  enum attachment_category_e { color = 0, depth_stencil, input, preserve };
  struct ZPC_CORE_API AttachmentDesc {
    AttachmentDesc(vk::Format format, vk::ImageLayout initialLayout, vk::ImageLayout finalLayout)
        : format{format}, initialLayout{initialLayout}, finalLayout{finalLayout} {}
    ~AttachmentDesc() = default;

    attachment_category_e category = color;
    vk::Format format{};
    vk::ImageLayout initialLayout{}, finalLayout{};
    vk::AttachmentLoadOp loadOp{vk::AttachmentLoadOp::eDontCare};
    vk::AttachmentStoreOp storeOp{vk::AttachmentStoreOp::eStore};
    vk::SampleCountFlagBits sampleBits{vk::SampleCountFlagBits::e1};
  };

  struct ZPC_CORE_API SubpassDesc {
    /// @note depth attachment and its ref always come last

    // a. check this setup first
    std::vector<u32> colorRefs{}, inputRefs{};
    int depthStencilRef{-1};  // -1: inactive, i: target depth/stencil ref

    int depthStencilResolveRef{-1};  // -1: inactive, i: target depth/stencil ref
    std::vector<u32> colorResolveRefs{};

    // directly used later for renderpass creation
    mutable std::vector<vk::AttachmentReference2> colorAttachRefs;
    mutable std::vector<vk::AttachmentReference2> colorResolveAttachRefs;
    mutable std::vector<vk::AttachmentReference2> depthAttachRefs;
    mutable vk::SubpassDescriptionDepthStencilResolve dsResolveProp;
    mutable std::vector<vk::AttachmentReference2> inputAttachRefs;

    vk::SubpassDescription2 resolve(std::vector<vk::AttachmentDescription2>& attachments) const {
      auto subpass
          = vk::SubpassDescription2{}.setPipelineBindPoint(vk::PipelineBindPoint::eGraphics);

      // color
      if (colorRefs.size()) {
        colorAttachRefs.resize(colorRefs.size());
        for (int i = 0; i != colorRefs.size(); ++i) {
          const u32 attachmentNo = colorRefs[i];
          // const auto& attachment = attachments[attachmentNo];
          auto ref
              = vk::AttachmentReference2{attachmentNo, vk::ImageLayout::eColorAttachmentOptimal,
                                         vk::ImageAspectFlagBits::eColor};
          colorAttachRefs[i] = ref;
        }
        subpass.setColorAttachmentCount((u32)colorAttachRefs.size())
            .setPColorAttachments(colorAttachRefs.data());
      }
      // color resolves
      if (colorResolveRefs.size()) {
        colorResolveAttachRefs.resize(colorResolveRefs.size());
        for (int i = 0; i != colorResolveRefs.size(); ++i) {
          const u32 attachmentNo = colorResolveRefs[i];
          // const auto& attachment = attachments[attachmentNo];
          auto ref
              = vk::AttachmentReference2{attachmentNo, vk::ImageLayout::eColorAttachmentOptimal,
                                         vk::ImageAspectFlagBits::eColor};
          colorResolveAttachRefs[i] = ref;
        }
        subpass.setResolveAttachments(colorResolveAttachRefs);
      }
      // input
      if (inputRefs.size()) {
        inputAttachRefs.resize(inputRefs.size());
        for (int i = 0; i != inputRefs.size(); ++i) {
          const u32 attachmentNo = inputRefs[i];
          const auto& attachment = attachments[attachmentNo];
          auto ref = vk::AttachmentReference2{attachmentNo, vk::ImageLayout::eShaderReadOnlyOptimal,
                                              deduce_image_format_aspect_flag(attachment.format)};
          inputAttachRefs[i] = ref;
        }
        subpass.setInputAttachmentCount((u32)inputAttachRefs.size())
            .setPInputAttachments(inputAttachRefs.data());
      }
      // depth stencil
      depthAttachRefs.reserve((depthStencilRef != -1 ? 1 : 0)
                              + (depthStencilResolveRef != -1 ? 1 : 0));
      if (depthStencilRef != -1) {
        const auto& attachment = attachments[depthStencilRef];
        depthAttachRefs.push_back(vk::AttachmentReference2{
            (u32)depthStencilRef, vk::ImageLayout::eDepthStencilAttachmentOptimal,
            deduce_image_format_aspect_flag(attachment.format)});
        subpass.setPDepthStencilAttachment(&depthAttachRefs.back());
      }

      if (depthStencilResolveRef != -1) {
        const auto& attachment = attachments[depthStencilResolveRef];
        depthAttachRefs.push_back(vk::AttachmentReference2{
            (u32)depthStencilResolveRef, vk::ImageLayout::eDepthStencilAttachmentOptimal,
            deduce_image_format_aspect_flag(attachment.format)});

        dsResolveProp.pDepthStencilResolveAttachment = &depthAttachRefs.back();
        dsResolveProp.depthResolveMode = vk::ResolveModeFlagBits::eSampleZero;
        dsResolveProp.stencilResolveMode = vk::ResolveModeFlagBits::eSampleZero;
        subpass.setPNext(&dsResolveProp);
      }

      return subpass;
    }
  };

  /// @note https://www.khronos.org/blog/streamlining-render-passes
  struct ZPC_CORE_API RenderPass {
    RenderPass(VulkanContext& ctx)
        : ctx{ctx}, renderpass{VK_NULL_HANDLE}, attachments{}, subpasses{} {}
    RenderPass(RenderPass&& o) noexcept
        : ctx{o.ctx},
          renderpass{o.renderpass},
          attachments{std::move(o.attachments)},
          subpasses{std::move(o.subpasses)} {
      o.renderpass = VK_NULL_HANDLE;
    }
    ~RenderPass() { ctx.device.destroyRenderPass(renderpass, nullptr, ctx.dispatcher); }

    vk::RenderPass operator*() const { return renderpass; }
    operator vk::RenderPass() const { return renderpass; }

  protected:
    friend struct VulkanContext;
    friend struct Swapchain;
    friend struct RenderPassBuilder;
    friend struct PipelineBuilder;

    VulkanContext& ctx;
    vk::RenderPass renderpass;
    // helpful info for pipeline build
    std::vector<AttachmentDesc> attachments;
    std::vector<SubpassDesc> subpasses;
  };

  struct ZPC_CORE_API RenderPassBuilder {
    RenderPassBuilder(VulkanContext& ctx) noexcept
        : ctx{ctx}, _attachments{}, _subpassCount{1}, _subpasses{}, _subpassDependencies{} {}
    ~RenderPassBuilder() = default;

    RenderPassBuilder& addAttachment(const AttachmentDesc& desc) {
      // could check [desc] validity here
      _attachments.push_back(desc);
      return *this;
    }
    RenderPassBuilder& addAttachment(vk::Format format = vk::Format::eR8G8B8A8Unorm,
                                     vk::ImageLayout initialLayout = vk::ImageLayout::eUndefined,
                                     vk::ImageLayout finalLayout
                                     = vk::ImageLayout::eColorAttachmentOptimal,
                                     bool clear = true,
                                     vk::SampleCountFlagBits numSamples
                                     = vk::SampleCountFlagBits::e1) {
      AttachmentDesc desc{format, initialLayout, finalLayout};
      desc.category = is_depth_stencil_format(format) ? depth_stencil : color;
      desc.sampleBits = numSamples;
      if (clear)
        desc.loadOp = vk::AttachmentLoadOp::eClear;
      else {
        desc.loadOp = initialLayout == vk::ImageLayout::eUndefined ? vk::AttachmentLoadOp::eDontCare
                                                                   : vk::AttachmentLoadOp::eLoad;
      }
      return addAttachment(desc);
    }
    RenderPassBuilder& addDepthAttachment(vk::Format format, bool clear,
                                          vk::SampleCountFlagBits numSamples
                                          = vk::SampleCountFlagBits::e1) {
      AttachmentDesc desc{format, vk::ImageLayout::eDepthStencilAttachmentOptimal,
                          vk::ImageLayout::eDepthStencilAttachmentOptimal};
      desc.category = depth_stencil;
      desc.sampleBits = numSamples;
      desc.loadOp = clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad;
      return addAttachment(desc);
    }

    RenderPassBuilder& addSubpass(/*color*/ const std::vector<u32>& colorRange,
                                  int depthStencilRef = -1,
                                  const std::vector<u32>& colorResolveRange = {},
                                  int depthStencilResolveRef = -1,
                                  /*input*/ const std::vector<u32>& inputRange = {}) {
      SubpassDesc sd;
      sd.colorRefs = colorRange;
      sd.inputRefs = inputRange;
      sd.depthStencilRef = depthStencilRef;
      sd.colorResolveRefs = colorResolveRange;
      sd.depthStencilResolveRef = depthStencilResolveRef;
      _subpasses.push_back(sd);
      return *this;
    }
    RenderPassBuilder& setSubpassDependencies(
        const std::vector<vk::SubpassDependency2>& subpassDependencies) {
      _subpassDependencies = subpassDependencies;
      return *this;
    }
    RenderPassBuilder& setNumPasses(u32 cnt) {
      _subpassCount = cnt;
      return *this;
    }

    RenderPass build() const {
      RenderPass ret{ctx};
      const u32 num = _attachments.size();
      std::vector<vk::AttachmentDescription2> attachments;
      attachments.reserve(num);

      std::vector<vk::AttachmentReference2> colorRefs;
      std::vector<vk::AttachmentReference2> depthRefs;
      vk::SubpassDescriptionDepthStencilResolve dsResolveProp;
      int dsRefIndex = -1, dsResolveRefIndex = -1;
      colorRefs.reserve(num);
      depthRefs.reserve(2);  // at most 2

      bool autoBuildRefs = _subpassDependencies.size() == 0 && _subpasses.size() == 0;
      SubpassDesc autoSubpassDesc;
      for (int i = 0; i != _attachments.size(); ++i) {
        const auto& attachmentDesc = _attachments[i];
        //
        attachments.push_back(vk::AttachmentDescription2{}
                                  .setFormat(attachmentDesc.format)
                                  .setSamples(attachmentDesc.sampleBits)
                                  .setLoadOp(attachmentDesc.loadOp)
                                  .setStoreOp(attachmentDesc.storeOp)
                                  .setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
                                  .setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
                                  .setInitialLayout(attachmentDesc.initialLayout)
                                  .setFinalLayout(attachmentDesc.finalLayout));
        //
        if (autoBuildRefs) {
          if (attachmentDesc.category == color) {
            colorRefs.push_back(
                vk::AttachmentReference2{(u32)i, vk::ImageLayout::eColorAttachmentOptimal,
                                         deduce_image_format_aspect_flag(attachmentDesc.format)});

            autoSubpassDesc.colorRefs.push_back(i);  // *

          } else if (attachmentDesc.category == depth_stencil) {
            depthRefs.push_back(
                vk::AttachmentReference2{(u32)i, vk::ImageLayout::eDepthStencilAttachmentOptimal,
                                         deduce_image_format_aspect_flag(attachmentDesc.format)});

            if (attachmentDesc.sampleBits == vk::SampleCountFlagBits::e1) {
              if (dsRefIndex == -1) {
                dsRefIndex = depthRefs.size() - 1;

                autoSubpassDesc.depthStencilRef = i;  // *
              } else
                throw std::runtime_error(
                    "there exists multiple depth stencil attachments with 1 sample bit!");
            } else {
              if (dsResolveRefIndex == -1) {
                dsResolveRefIndex = depthRefs.size() - 1;

                autoSubpassDesc.depthStencilRef = i;  // *
              } else
                throw std::runtime_error("there exists multiple msaa depth stencil attachments!");
              dsResolveProp.pDepthStencilResolveAttachment = &depthRefs[dsResolveRefIndex];
              dsResolveProp.depthResolveMode = vk::ResolveModeFlagBits::eSampleZero;
              dsResolveProp.stencilResolveMode = vk::ResolveModeFlagBits::eSampleZero;
            }
          } else
            throw std::runtime_error("currently do not support rp input attachments");
        }
      }

      if (autoBuildRefs) {
        std::vector<vk::SubpassDescription2> subpasses;
        std::vector<vk::SubpassDependency2> subpassDependencies;

        vk::AccessFlags accessFlag = vk::AccessFlagBits::eColorAttachmentWrite;
        if (dsRefIndex != -1) accessFlag |= vk::AccessFlagBits::eDepthStencilAttachmentWrite;

        auto subpass = vk::SubpassDescription2{}
                           .setPipelineBindPoint(vk::PipelineBindPoint::eGraphics)
                           .setColorAttachmentCount((u32)colorRefs.size())
                           .setPColorAttachments(colorRefs.data());
        if (dsRefIndex != -1) subpass.setPDepthStencilAttachment(&depthRefs[dsRefIndex]);
        if (dsResolveRefIndex != -1) subpass.setPNext(&dsResolveProp);
        subpasses.resize(_subpassCount, subpass);

        for (u32 i = 0; i < _subpassCount; i++) {
          auto dependency = vk::SubpassDependency2{}
                                .setSrcSubpass(i == 0 ? (VK_SUBPASS_EXTERNAL) : (i - 1))
                                .setDstSubpass(i)
                                .setSrcAccessMask({})
                                .setDstAccessMask(accessFlag)
                                .setSrcStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput
                                                 | vk::PipelineStageFlagBits::eLateFragmentTests)
                                .setDstStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput
                                                 | vk::PipelineStageFlagBits::eEarlyFragmentTests);

          subpassDependencies.push_back(dependency);
        }
        ret.renderpass
            = ctx.device.createRenderPass2(vk::RenderPassCreateInfo2{}
                                               .setAttachmentCount(attachments.size())
                                               .setPAttachments(attachments.data())
                                               .setSubpassCount(subpasses.size())
                                               .setPSubpasses(subpasses.data())
                                               .setDependencyCount(subpassDependencies.size())
                                               .setPDependencies(subpassDependencies.data()),
                                           nullptr, ctx.dispatcher);

        ret.subpasses = std::vector<SubpassDesc>(_subpassCount, autoSubpassDesc);
      } else {
        std::vector<vk::SubpassDescription2> subpasses(_subpasses.size());
        for (u32 i = 0; i < subpasses.size(); i++)
          subpasses[i] = _subpasses[i].resolve(attachments);

        ret.renderpass
            = ctx.device.createRenderPass2(vk::RenderPassCreateInfo2{}
                                               .setAttachmentCount(attachments.size())
                                               .setPAttachments(attachments.data())
                                               .setSubpassCount(subpasses.size())
                                               .setPSubpasses(subpasses.data())
                                               .setDependencyCount(_subpassDependencies.size())
                                               .setPDependencies(_subpassDependencies.data()),
                                           nullptr, ctx.dispatcher);

        ret.subpasses = _subpasses;
      }

      ret.attachments = _attachments;

      return ret;
    }

  private:
    VulkanContext& ctx;

    std::vector<AttachmentDesc> _attachments;

    // a
    std::vector<SubpassDesc> _subpasses;
    std::vector<vk::SubpassDependency2> _subpassDependencies;
    // b
    u32 _subpassCount;
  };

}  // namespace zs