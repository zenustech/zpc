#include "zensim/vulkan/VkShader.hpp"

#include <shaderc/shaderc.hpp>
#include <spirv_cross/spirv_glsl.hpp>

#include "zensim/vulkan/Vulkan.hpp"

namespace zs {

  static std::string reflect_basetype_string(spirv_cross::SPIRType::BaseType t) {
    switch (t) {
      case spirv_cross::SPIRType::Unknown:
        return "unknown";
      case spirv_cross::SPIRType::Void:
        return "void";
      case spirv_cross::SPIRType::Boolean:
        return "boolean";
      case spirv_cross::SPIRType::SByte:
        return "signed byte";
      case spirv_cross::SPIRType::UByte:
        return "unsigned byte";
      case spirv_cross::SPIRType::Short:
        return "short";
      case spirv_cross::SPIRType::UShort:
        return "unsigned short";
      case spirv_cross::SPIRType::Int:
        return "int";
      case spirv_cross::SPIRType::UInt:
        return "unsigned int";
      case spirv_cross::SPIRType::Int64:
        return "int64";
      case spirv_cross::SPIRType::UInt64:
        return "unsigned int64";
      case spirv_cross::SPIRType::AtomicCounter:
        return "atomic counter";
      case spirv_cross::SPIRType::Half:
        return "half";
      case spirv_cross::SPIRType::Float:
        return "float";
      case spirv_cross::SPIRType::Double:
        return "double";
      case spirv_cross::SPIRType::Struct:
        return "struct";
      case spirv_cross::SPIRType::Image:
        return "image";
      case spirv_cross::SPIRType::SampledImage:
        return "sampled image";
      case spirv_cross::SPIRType::Sampler:
        return "sampler";

      case spirv_cross::SPIRType::Char:
        return "char";
      default:;
    }
    return "wtf type";
  }
  static vk::Format reflect_basetype_vkformat(spirv_cross::SPIRType::BaseType t, u32 dim) {
    switch (t) {
      case spirv_cross::SPIRType::Boolean:
        return deduce_attribute_format(wrapt<bool>{}, dim);
      case spirv_cross::SPIRType::SByte:
        return deduce_attribute_format(wrapt<i8>{}, dim);
      case spirv_cross::SPIRType::UByte:
        return deduce_attribute_format(wrapt<u8>{}, dim);
      case spirv_cross::SPIRType::Short:
        return deduce_attribute_format(wrapt<short>{}, dim);
      case spirv_cross::SPIRType::UShort:
        return deduce_attribute_format(wrapt<unsigned short>{}, dim);
      case spirv_cross::SPIRType::Int:
        return deduce_attribute_format(wrapt<int>{}, dim);
      case spirv_cross::SPIRType::UInt:
        return deduce_attribute_format(wrapt<unsigned int>{}, dim);
      case spirv_cross::SPIRType::Int64:
        return deduce_attribute_format(wrapt<i64>{}, dim);
      case spirv_cross::SPIRType::UInt64:
        return deduce_attribute_format(wrapt<u64>{}, dim);
      // case spirv_cross::SPIRType::Half:
      //  return deduce_attribute_format(wrapt<i64>{}, dim);
      case spirv_cross::SPIRType::Float:
        return deduce_attribute_format(wrapt<float>{}, dim);
      case spirv_cross::SPIRType::Double:
        return deduce_attribute_format(wrapt<double>{}, dim);
      case spirv_cross::SPIRType::Char:
        return deduce_attribute_format(wrapt<char>{}, dim);
      default:;
    }
    return vk::Format::eUndefined;
  }

  static void display_resource(const spirv_cross::CompilerGLSL &glsl,
                               const spirv_cross::ShaderResources &resources) {
    auto displayBindingInfo = [&glsl](const auto &resources, std::string_view tag) {
      for (auto &resource : resources) {
        // spirv-cross/spirv_common.hpp spirv_cross.hpp main.cpp
        unsigned set = glsl.get_decoration(resource.id, spv::DecorationDescriptorSet);
        unsigned binding = glsl.get_decoration(resource.id, spv::DecorationBinding);
        unsigned location = glsl.get_decoration(resource.id, spv::DecorationLocation);

        const spirv_cross::SPIRType &type = glsl.get_type(resource.type_id);
        u32 typeArraySize = type.array.size();
        u32 count = typeArraySize == 0 ? 1 : type.array[0];
        auto typestr = reflect_basetype_string(type.basetype);

        fmt::print(
            "[{}] {} at set = {}, binding = {}, location = {}, basetype: {} (dim: width [{}], "
            "vecsize[{}], cols[{}]), typeArraySize: {}, count(s): {}",
            tag, resource.name.c_str(), set, binding, location, typestr, type.width, type.vecsize,
            type.columns, typeArraySize, count);
        for (int j = 1; j < typeArraySize; ++j) fmt::print(", {}", type.array[j]);
        fmt::print(". (inner->outer)\n");
      }
    };
    fmt::print("\n========displaying shader resource reflection========\n");
    displayBindingInfo(resources.uniform_buffers, "uniform buffer");
    displayBindingInfo(resources.storage_buffers, "storage buffer");
    displayBindingInfo(resources.stage_inputs, "stage inputs");
    displayBindingInfo(resources.stage_outputs, "stage outputs");
    displayBindingInfo(resources.subpass_inputs, "subpass inputs");
    displayBindingInfo(resources.storage_images, "storage images");
    displayBindingInfo(resources.sampled_images, "sampled images");
    displayBindingInfo(resources.atomic_counters, "atomic counters");
    displayBindingInfo(resources.acceleration_structures, "acceleration structures");
    displayBindingInfo(resources.push_constant_buffers, "push constant buffers");
    displayBindingInfo(resources.shader_record_buffers, "shader record buffers");
    displayBindingInfo(resources.separate_images, "separate images");
    displayBindingInfo(resources.separate_samplers, "separate samplers");
    fmt::print("=====================================================\n\n");
  }

  void ShaderModule::analyzeLayout(const u32 *code, size_t size) {
    // check spirv_parse.cpp for the meaning of 'size': word_count
    compiled = std::unique_ptr<void, void (*)(void const *)>(
        new spirv_cross::CompilerGLSL(code, size),
        [](void const *data) { delete static_cast<spirv_cross::CompilerGLSL const *>(data); });

    resources = std::unique_ptr<void, void (*)(void const *)>(
        new spirv_cross::ShaderResources(
            static_cast<spirv_cross::CompilerGLSL *>(compiled.get())->get_shader_resources()),
        [](void const *data) { delete static_cast<spirv_cross::ShaderResources const *>(data); });
  }

  void ShaderModule::initializeDescriptorSetLayouts() {
    setLayouts.clear();
    auto &glsl = *static_cast<spirv_cross::CompilerGLSL *>(compiled.get());
    auto &resources_ = *static_cast<spirv_cross::ShaderResources *>(resources.get());
    auto generateDescriptors = [&glsl, this](const auto &resources,
                                             vk::DescriptorType descriptorType) {
      for (auto &resource : resources) {
        u32 set = glsl.get_decoration(resource.id, spv::DecorationDescriptorSet);
        u32 binding = glsl.get_decoration(resource.id, spv::DecorationBinding);
        u32 location = glsl.get_decoration(resource.id, spv::DecorationLocation);

        fmt::print(
            "---->\tadding descriptor set layout {} at set [{}], binding [{}], location [{}]\n",
            resource.name.c_str(), set, binding, location);
        setLayouts.emplace(
            set, ctx.setlayout().addBinding(binding, descriptorType, stageFlag, 1).build());
      }
    };
    generateDescriptors(resources_.uniform_buffers, vk::DescriptorType::eUniformBufferDynamic);
    generateDescriptors(resources_.storage_buffers, vk::DescriptorType::eStorageBuffer);
    generateDescriptors(resources_.storage_images, vk::DescriptorType::eStorageImage);
    generateDescriptors(resources_.sampled_images, vk::DescriptorType::eCombinedImageSampler);
  }

  void ShaderModule::initializeInputAttributes() {
    inputAttributes.clear();
    if (stageFlag != vk::ShaderStageFlagBits::eVertex) return;
    // std::map<u32, std::map<u32, AttributeDescriptor>>
    auto &glsl = *static_cast<spirv_cross::CompilerGLSL *>(compiled.get());
    auto &resources_ = *static_cast<spirv_cross::ShaderResources *>(resources.get());
    auto generateInputDescriptions = [&glsl, this](const auto &resources) {
      for (auto &resource : resources) {
        // u32 set = glsl.get_decoration(resource.id, spv::DecorationDescriptorSet);
        // u32 binding = glsl.get_decoration(resource.id, spv::DecorationBinding);
        u32 location = glsl.get_decoration(resource.id, spv::DecorationLocation);

        const spirv_cross::SPIRType &type = glsl.get_type(resource.type_id);
        u32 typeArraySize = type.array.size();
        u32 count = typeArraySize == 0 ? 1 : type.array[0];
        auto typestr = reflect_basetype_string(type.basetype);

        std::vector<u32> dims(typeArraySize);
        for (int j = 0; j < typeArraySize; ++j) dims[j] = type.array[j];

        fmt::print(
            "{} at location = {}, basetype: {} (dim: width [{}], "
            "vecsize[{}], cols[{}]), typeArraySize: {}, count(s): {}",
            resource.name.c_str(), location, typestr, type.width, type.vecsize, type.columns,
            typeArraySize, count);
        for (int j = 1; j < typeArraySize; ++j) fmt::print(", {}", type.array[j]);
        fmt::print(". (inner->outer)\n");

        /// @note num rows: vecsize; num cols: columns
        auto format = reflect_basetype_vkformat(type.basetype, type.vecsize);
        if (format == vk::Format::eUndefined)
          throw std::runtime_error(
              fmt::format("no appropriate vkformat duduced for this (column) type: <{}, {}>",
                          typestr, type.vecsize));
        inputAttributes[location]
            = AttributeDescriptor{/*alignment bits*/ type.width,
                                  /*size*/ (u32)(type.vecsize * type.columns * type.width / 8),
                                  /*format*/ format, std::move(dims)};
      }
    };
    generateInputDescriptions(resources_.stage_inputs);
  }

  void ShaderModule::displayLayoutInfo() {
    display_resource(*static_cast<spirv_cross::CompilerGLSL *>(compiled.get()),
                     *static_cast<spirv_cross::ShaderResources *>(resources.get()));
  }

}  // namespace zs