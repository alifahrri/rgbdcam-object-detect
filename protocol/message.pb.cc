// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: message.proto

#define INTERNAL_SUPPRESS_PROTOBUF_FIELD_DEPRECATION
#include "message.pb.h"

#include <algorithm>

#include <google/protobuf/stubs/common.h>
#include <google/protobuf/stubs/once.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/wire_format_lite_inl.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/reflection_ops.h>
#include <google/protobuf/wire_format.h>
// @@protoc_insertion_point(includes)

namespace object_detection {

namespace {

const ::google::protobuf::Descriptor* Detections_descriptor_ = NULL;
const ::google::protobuf::internal::GeneratedMessageReflection*
  Detections_reflection_ = NULL;
const ::google::protobuf::Descriptor* Detections_Object_descriptor_ = NULL;
const ::google::protobuf::internal::GeneratedMessageReflection*
  Detections_Object_reflection_ = NULL;

}  // namespace


void protobuf_AssignDesc_message_2eproto() {
  protobuf_AddDesc_message_2eproto();
  const ::google::protobuf::FileDescriptor* file =
    ::google::protobuf::DescriptorPool::generated_pool()->FindFileByName(
      "message.proto");
  GOOGLE_CHECK(file != NULL);
  Detections_descriptor_ = file->message_type(0);
  static const int Detections_offsets_[1] = {
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(Detections, objects_),
  };
  Detections_reflection_ =
    new ::google::protobuf::internal::GeneratedMessageReflection(
      Detections_descriptor_,
      Detections::default_instance_,
      Detections_offsets_,
      GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(Detections, _has_bits_[0]),
      GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(Detections, _unknown_fields_),
      -1,
      ::google::protobuf::DescriptorPool::generated_pool(),
      ::google::protobuf::MessageFactory::generated_factory(),
      sizeof(Detections));
  Detections_Object_descriptor_ = Detections_descriptor_->nested_type(0);
  static const int Detections_Object_offsets_[8] = {
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(Detections_Object, class__),
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(Detections_Object, x_pixel_),
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(Detections_Object, y_pixel_),
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(Detections_Object, z_pixel_),
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(Detections_Object, x_mm_),
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(Detections_Object, y_mm_),
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(Detections_Object, z_mm_),
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(Detections_Object, confidence_),
  };
  Detections_Object_reflection_ =
    new ::google::protobuf::internal::GeneratedMessageReflection(
      Detections_Object_descriptor_,
      Detections_Object::default_instance_,
      Detections_Object_offsets_,
      GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(Detections_Object, _has_bits_[0]),
      GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(Detections_Object, _unknown_fields_),
      -1,
      ::google::protobuf::DescriptorPool::generated_pool(),
      ::google::protobuf::MessageFactory::generated_factory(),
      sizeof(Detections_Object));
}

namespace {

GOOGLE_PROTOBUF_DECLARE_ONCE(protobuf_AssignDescriptors_once_);
inline void protobuf_AssignDescriptorsOnce() {
  ::google::protobuf::GoogleOnceInit(&protobuf_AssignDescriptors_once_,
                 &protobuf_AssignDesc_message_2eproto);
}

void protobuf_RegisterTypes(const ::std::string&) {
  protobuf_AssignDescriptorsOnce();
  ::google::protobuf::MessageFactory::InternalRegisterGeneratedMessage(
    Detections_descriptor_, &Detections::default_instance());
  ::google::protobuf::MessageFactory::InternalRegisterGeneratedMessage(
    Detections_Object_descriptor_, &Detections_Object::default_instance());
}

}  // namespace

void protobuf_ShutdownFile_message_2eproto() {
  delete Detections::default_instance_;
  delete Detections_reflection_;
  delete Detections_Object::default_instance_;
  delete Detections_Object_reflection_;
}

void protobuf_AddDesc_message_2eproto() {
  static bool already_here = false;
  if (already_here) return;
  already_here = true;
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  ::google::protobuf::DescriptorPool::InternalAddGeneratedFile(
    "\n\rmessage.proto\022\020object_detection\"\315\001\n\nDe"
    "tections\0224\n\007objects\030\001 \003(\0132#.object_detec"
    "tion.Detections.Object\032\210\001\n\006Object\022\r\n\005cla"
    "ss\030\001 \002(\t\022\017\n\007x_pixel\030\002 \002(\002\022\017\n\007y_pixel\030\003 \002"
    "(\002\022\017\n\007z_pixel\030\004 \002(\002\022\014\n\004x_mm\030\005 \002(\002\022\014\n\004y_m"
    "m\030\006 \002(\002\022\014\n\004z_mm\030\007 \002(\002\022\022\n\nconfidence\030\010 \002("
    "\002", 241);
  ::google::protobuf::MessageFactory::InternalRegisterGeneratedFile(
    "message.proto", &protobuf_RegisterTypes);
  Detections::default_instance_ = new Detections();
  Detections_Object::default_instance_ = new Detections_Object();
  Detections::default_instance_->InitAsDefaultInstance();
  Detections_Object::default_instance_->InitAsDefaultInstance();
  ::google::protobuf::internal::OnShutdown(&protobuf_ShutdownFile_message_2eproto);
}

// Force AddDescriptors() to be called at static initialization time.
struct StaticDescriptorInitializer_message_2eproto {
  StaticDescriptorInitializer_message_2eproto() {
    protobuf_AddDesc_message_2eproto();
  }
} static_descriptor_initializer_message_2eproto_;

// ===================================================================

#ifndef _MSC_VER
const int Detections_Object::kClassFieldNumber;
const int Detections_Object::kXPixelFieldNumber;
const int Detections_Object::kYPixelFieldNumber;
const int Detections_Object::kZPixelFieldNumber;
const int Detections_Object::kXMmFieldNumber;
const int Detections_Object::kYMmFieldNumber;
const int Detections_Object::kZMmFieldNumber;
const int Detections_Object::kConfidenceFieldNumber;
#endif  // !_MSC_VER

Detections_Object::Detections_Object()
  : ::google::protobuf::Message() {
  SharedCtor();
  // @@protoc_insertion_point(constructor:object_detection.Detections.Object)
}

void Detections_Object::InitAsDefaultInstance() {
}

Detections_Object::Detections_Object(const Detections_Object& from)
  : ::google::protobuf::Message() {
  SharedCtor();
  MergeFrom(from);
  // @@protoc_insertion_point(copy_constructor:object_detection.Detections.Object)
}

void Detections_Object::SharedCtor() {
  ::google::protobuf::internal::GetEmptyString();
  _cached_size_ = 0;
  class__ = const_cast< ::std::string*>(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  x_pixel_ = 0;
  y_pixel_ = 0;
  z_pixel_ = 0;
  x_mm_ = 0;
  y_mm_ = 0;
  z_mm_ = 0;
  confidence_ = 0;
  ::memset(_has_bits_, 0, sizeof(_has_bits_));
}

Detections_Object::~Detections_Object() {
  // @@protoc_insertion_point(destructor:object_detection.Detections.Object)
  SharedDtor();
}

void Detections_Object::SharedDtor() {
  if (class__ != &::google::protobuf::internal::GetEmptyStringAlreadyInited()) {
    delete class__;
  }
  if (this != default_instance_) {
  }
}

void Detections_Object::SetCachedSize(int size) const {
  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
  _cached_size_ = size;
  GOOGLE_SAFE_CONCURRENT_WRITES_END();
}
const ::google::protobuf::Descriptor* Detections_Object::descriptor() {
  protobuf_AssignDescriptorsOnce();
  return Detections_Object_descriptor_;
}

const Detections_Object& Detections_Object::default_instance() {
  if (default_instance_ == NULL) protobuf_AddDesc_message_2eproto();
  return *default_instance_;
}

Detections_Object* Detections_Object::default_instance_ = NULL;

Detections_Object* Detections_Object::New() const {
  return new Detections_Object;
}

void Detections_Object::Clear() {
#define OFFSET_OF_FIELD_(f) (reinterpret_cast<char*>(      \
  &reinterpret_cast<Detections_Object*>(16)->f) - \
   reinterpret_cast<char*>(16))

#define ZR_(first, last) do {                              \
    size_t f = OFFSET_OF_FIELD_(first);                    \
    size_t n = OFFSET_OF_FIELD_(last) - f + sizeof(last);  \
    ::memset(&first, 0, n);                                \
  } while (0)

  if (_has_bits_[0 / 32] & 255) {
    ZR_(x_pixel_, confidence_);
    if (has_class_()) {
      if (class__ != &::google::protobuf::internal::GetEmptyStringAlreadyInited()) {
        class__->clear();
      }
    }
  }

#undef OFFSET_OF_FIELD_
#undef ZR_

  ::memset(_has_bits_, 0, sizeof(_has_bits_));
  mutable_unknown_fields()->Clear();
}

bool Detections_Object::MergePartialFromCodedStream(
    ::google::protobuf::io::CodedInputStream* input) {
#define DO_(EXPRESSION) if (!(EXPRESSION)) goto failure
  ::google::protobuf::uint32 tag;
  // @@protoc_insertion_point(parse_start:object_detection.Detections.Object)
  for (;;) {
    ::std::pair< ::google::protobuf::uint32, bool> p = input->ReadTagWithCutoff(127);
    tag = p.first;
    if (!p.second) goto handle_unusual;
    switch (::google::protobuf::internal::WireFormatLite::GetTagFieldNumber(tag)) {
      // required string class = 1;
      case 1: {
        if (tag == 10) {
          DO_(::google::protobuf::internal::WireFormatLite::ReadString(
                input, this->mutable_class_()));
          ::google::protobuf::internal::WireFormat::VerifyUTF8StringNamedField(
            this->class_().data(), this->class_().length(),
            ::google::protobuf::internal::WireFormat::PARSE,
            "class_");
        } else {
          goto handle_unusual;
        }
        if (input->ExpectTag(21)) goto parse_x_pixel;
        break;
      }

      // required float x_pixel = 2;
      case 2: {
        if (tag == 21) {
         parse_x_pixel:
          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   float, ::google::protobuf::internal::WireFormatLite::TYPE_FLOAT>(
                 input, &x_pixel_)));
          set_has_x_pixel();
        } else {
          goto handle_unusual;
        }
        if (input->ExpectTag(29)) goto parse_y_pixel;
        break;
      }

      // required float y_pixel = 3;
      case 3: {
        if (tag == 29) {
         parse_y_pixel:
          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   float, ::google::protobuf::internal::WireFormatLite::TYPE_FLOAT>(
                 input, &y_pixel_)));
          set_has_y_pixel();
        } else {
          goto handle_unusual;
        }
        if (input->ExpectTag(37)) goto parse_z_pixel;
        break;
      }

      // required float z_pixel = 4;
      case 4: {
        if (tag == 37) {
         parse_z_pixel:
          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   float, ::google::protobuf::internal::WireFormatLite::TYPE_FLOAT>(
                 input, &z_pixel_)));
          set_has_z_pixel();
        } else {
          goto handle_unusual;
        }
        if (input->ExpectTag(45)) goto parse_x_mm;
        break;
      }

      // required float x_mm = 5;
      case 5: {
        if (tag == 45) {
         parse_x_mm:
          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   float, ::google::protobuf::internal::WireFormatLite::TYPE_FLOAT>(
                 input, &x_mm_)));
          set_has_x_mm();
        } else {
          goto handle_unusual;
        }
        if (input->ExpectTag(53)) goto parse_y_mm;
        break;
      }

      // required float y_mm = 6;
      case 6: {
        if (tag == 53) {
         parse_y_mm:
          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   float, ::google::protobuf::internal::WireFormatLite::TYPE_FLOAT>(
                 input, &y_mm_)));
          set_has_y_mm();
        } else {
          goto handle_unusual;
        }
        if (input->ExpectTag(61)) goto parse_z_mm;
        break;
      }

      // required float z_mm = 7;
      case 7: {
        if (tag == 61) {
         parse_z_mm:
          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   float, ::google::protobuf::internal::WireFormatLite::TYPE_FLOAT>(
                 input, &z_mm_)));
          set_has_z_mm();
        } else {
          goto handle_unusual;
        }
        if (input->ExpectTag(69)) goto parse_confidence;
        break;
      }

      // required float confidence = 8;
      case 8: {
        if (tag == 69) {
         parse_confidence:
          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   float, ::google::protobuf::internal::WireFormatLite::TYPE_FLOAT>(
                 input, &confidence_)));
          set_has_confidence();
        } else {
          goto handle_unusual;
        }
        if (input->ExpectAtEnd()) goto success;
        break;
      }

      default: {
      handle_unusual:
        if (tag == 0 ||
            ::google::protobuf::internal::WireFormatLite::GetTagWireType(tag) ==
            ::google::protobuf::internal::WireFormatLite::WIRETYPE_END_GROUP) {
          goto success;
        }
        DO_(::google::protobuf::internal::WireFormat::SkipField(
              input, tag, mutable_unknown_fields()));
        break;
      }
    }
  }
success:
  // @@protoc_insertion_point(parse_success:object_detection.Detections.Object)
  return true;
failure:
  // @@protoc_insertion_point(parse_failure:object_detection.Detections.Object)
  return false;
#undef DO_
}

void Detections_Object::SerializeWithCachedSizes(
    ::google::protobuf::io::CodedOutputStream* output) const {
  // @@protoc_insertion_point(serialize_start:object_detection.Detections.Object)
  // required string class = 1;
  if (has_class_()) {
    ::google::protobuf::internal::WireFormat::VerifyUTF8StringNamedField(
      this->class_().data(), this->class_().length(),
      ::google::protobuf::internal::WireFormat::SERIALIZE,
      "class_");
    ::google::protobuf::internal::WireFormatLite::WriteStringMaybeAliased(
      1, this->class_(), output);
  }

  // required float x_pixel = 2;
  if (has_x_pixel()) {
    ::google::protobuf::internal::WireFormatLite::WriteFloat(2, this->x_pixel(), output);
  }

  // required float y_pixel = 3;
  if (has_y_pixel()) {
    ::google::protobuf::internal::WireFormatLite::WriteFloat(3, this->y_pixel(), output);
  }

  // required float z_pixel = 4;
  if (has_z_pixel()) {
    ::google::protobuf::internal::WireFormatLite::WriteFloat(4, this->z_pixel(), output);
  }

  // required float x_mm = 5;
  if (has_x_mm()) {
    ::google::protobuf::internal::WireFormatLite::WriteFloat(5, this->x_mm(), output);
  }

  // required float y_mm = 6;
  if (has_y_mm()) {
    ::google::protobuf::internal::WireFormatLite::WriteFloat(6, this->y_mm(), output);
  }

  // required float z_mm = 7;
  if (has_z_mm()) {
    ::google::protobuf::internal::WireFormatLite::WriteFloat(7, this->z_mm(), output);
  }

  // required float confidence = 8;
  if (has_confidence()) {
    ::google::protobuf::internal::WireFormatLite::WriteFloat(8, this->confidence(), output);
  }

  if (!unknown_fields().empty()) {
    ::google::protobuf::internal::WireFormat::SerializeUnknownFields(
        unknown_fields(), output);
  }
  // @@protoc_insertion_point(serialize_end:object_detection.Detections.Object)
}

::google::protobuf::uint8* Detections_Object::SerializeWithCachedSizesToArray(
    ::google::protobuf::uint8* target) const {
  // @@protoc_insertion_point(serialize_to_array_start:object_detection.Detections.Object)
  // required string class = 1;
  if (has_class_()) {
    ::google::protobuf::internal::WireFormat::VerifyUTF8StringNamedField(
      this->class_().data(), this->class_().length(),
      ::google::protobuf::internal::WireFormat::SERIALIZE,
      "class_");
    target =
      ::google::protobuf::internal::WireFormatLite::WriteStringToArray(
        1, this->class_(), target);
  }

  // required float x_pixel = 2;
  if (has_x_pixel()) {
    target = ::google::protobuf::internal::WireFormatLite::WriteFloatToArray(2, this->x_pixel(), target);
  }

  // required float y_pixel = 3;
  if (has_y_pixel()) {
    target = ::google::protobuf::internal::WireFormatLite::WriteFloatToArray(3, this->y_pixel(), target);
  }

  // required float z_pixel = 4;
  if (has_z_pixel()) {
    target = ::google::protobuf::internal::WireFormatLite::WriteFloatToArray(4, this->z_pixel(), target);
  }

  // required float x_mm = 5;
  if (has_x_mm()) {
    target = ::google::protobuf::internal::WireFormatLite::WriteFloatToArray(5, this->x_mm(), target);
  }

  // required float y_mm = 6;
  if (has_y_mm()) {
    target = ::google::protobuf::internal::WireFormatLite::WriteFloatToArray(6, this->y_mm(), target);
  }

  // required float z_mm = 7;
  if (has_z_mm()) {
    target = ::google::protobuf::internal::WireFormatLite::WriteFloatToArray(7, this->z_mm(), target);
  }

  // required float confidence = 8;
  if (has_confidence()) {
    target = ::google::protobuf::internal::WireFormatLite::WriteFloatToArray(8, this->confidence(), target);
  }

  if (!unknown_fields().empty()) {
    target = ::google::protobuf::internal::WireFormat::SerializeUnknownFieldsToArray(
        unknown_fields(), target);
  }
  // @@protoc_insertion_point(serialize_to_array_end:object_detection.Detections.Object)
  return target;
}

int Detections_Object::ByteSize() const {
  int total_size = 0;

  if (_has_bits_[0 / 32] & (0xffu << (0 % 32))) {
    // required string class = 1;
    if (has_class_()) {
      total_size += 1 +
        ::google::protobuf::internal::WireFormatLite::StringSize(
          this->class_());
    }

    // required float x_pixel = 2;
    if (has_x_pixel()) {
      total_size += 1 + 4;
    }

    // required float y_pixel = 3;
    if (has_y_pixel()) {
      total_size += 1 + 4;
    }

    // required float z_pixel = 4;
    if (has_z_pixel()) {
      total_size += 1 + 4;
    }

    // required float x_mm = 5;
    if (has_x_mm()) {
      total_size += 1 + 4;
    }

    // required float y_mm = 6;
    if (has_y_mm()) {
      total_size += 1 + 4;
    }

    // required float z_mm = 7;
    if (has_z_mm()) {
      total_size += 1 + 4;
    }

    // required float confidence = 8;
    if (has_confidence()) {
      total_size += 1 + 4;
    }

  }
  if (!unknown_fields().empty()) {
    total_size +=
      ::google::protobuf::internal::WireFormat::ComputeUnknownFieldsSize(
        unknown_fields());
  }
  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
  _cached_size_ = total_size;
  GOOGLE_SAFE_CONCURRENT_WRITES_END();
  return total_size;
}

void Detections_Object::MergeFrom(const ::google::protobuf::Message& from) {
  GOOGLE_CHECK_NE(&from, this);
  const Detections_Object* source =
    ::google::protobuf::internal::dynamic_cast_if_available<const Detections_Object*>(
      &from);
  if (source == NULL) {
    ::google::protobuf::internal::ReflectionOps::Merge(from, this);
  } else {
    MergeFrom(*source);
  }
}

void Detections_Object::MergeFrom(const Detections_Object& from) {
  GOOGLE_CHECK_NE(&from, this);
  if (from._has_bits_[0 / 32] & (0xffu << (0 % 32))) {
    if (from.has_class_()) {
      set_class_(from.class_());
    }
    if (from.has_x_pixel()) {
      set_x_pixel(from.x_pixel());
    }
    if (from.has_y_pixel()) {
      set_y_pixel(from.y_pixel());
    }
    if (from.has_z_pixel()) {
      set_z_pixel(from.z_pixel());
    }
    if (from.has_x_mm()) {
      set_x_mm(from.x_mm());
    }
    if (from.has_y_mm()) {
      set_y_mm(from.y_mm());
    }
    if (from.has_z_mm()) {
      set_z_mm(from.z_mm());
    }
    if (from.has_confidence()) {
      set_confidence(from.confidence());
    }
  }
  mutable_unknown_fields()->MergeFrom(from.unknown_fields());
}

void Detections_Object::CopyFrom(const ::google::protobuf::Message& from) {
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void Detections_Object::CopyFrom(const Detections_Object& from) {
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool Detections_Object::IsInitialized() const {
  if ((_has_bits_[0] & 0x000000ff) != 0x000000ff) return false;

  return true;
}

void Detections_Object::Swap(Detections_Object* other) {
  if (other != this) {
    std::swap(class__, other->class__);
    std::swap(x_pixel_, other->x_pixel_);
    std::swap(y_pixel_, other->y_pixel_);
    std::swap(z_pixel_, other->z_pixel_);
    std::swap(x_mm_, other->x_mm_);
    std::swap(y_mm_, other->y_mm_);
    std::swap(z_mm_, other->z_mm_);
    std::swap(confidence_, other->confidence_);
    std::swap(_has_bits_[0], other->_has_bits_[0]);
    _unknown_fields_.Swap(&other->_unknown_fields_);
    std::swap(_cached_size_, other->_cached_size_);
  }
}

::google::protobuf::Metadata Detections_Object::GetMetadata() const {
  protobuf_AssignDescriptorsOnce();
  ::google::protobuf::Metadata metadata;
  metadata.descriptor = Detections_Object_descriptor_;
  metadata.reflection = Detections_Object_reflection_;
  return metadata;
}


// -------------------------------------------------------------------

#ifndef _MSC_VER
const int Detections::kObjectsFieldNumber;
#endif  // !_MSC_VER

Detections::Detections()
  : ::google::protobuf::Message() {
  SharedCtor();
  // @@protoc_insertion_point(constructor:object_detection.Detections)
}

void Detections::InitAsDefaultInstance() {
}

Detections::Detections(const Detections& from)
  : ::google::protobuf::Message() {
  SharedCtor();
  MergeFrom(from);
  // @@protoc_insertion_point(copy_constructor:object_detection.Detections)
}

void Detections::SharedCtor() {
  _cached_size_ = 0;
  ::memset(_has_bits_, 0, sizeof(_has_bits_));
}

Detections::~Detections() {
  // @@protoc_insertion_point(destructor:object_detection.Detections)
  SharedDtor();
}

void Detections::SharedDtor() {
  if (this != default_instance_) {
  }
}

void Detections::SetCachedSize(int size) const {
  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
  _cached_size_ = size;
  GOOGLE_SAFE_CONCURRENT_WRITES_END();
}
const ::google::protobuf::Descriptor* Detections::descriptor() {
  protobuf_AssignDescriptorsOnce();
  return Detections_descriptor_;
}

const Detections& Detections::default_instance() {
  if (default_instance_ == NULL) protobuf_AddDesc_message_2eproto();
  return *default_instance_;
}

Detections* Detections::default_instance_ = NULL;

Detections* Detections::New() const {
  return new Detections;
}

void Detections::Clear() {
  objects_.Clear();
  ::memset(_has_bits_, 0, sizeof(_has_bits_));
  mutable_unknown_fields()->Clear();
}

bool Detections::MergePartialFromCodedStream(
    ::google::protobuf::io::CodedInputStream* input) {
#define DO_(EXPRESSION) if (!(EXPRESSION)) goto failure
  ::google::protobuf::uint32 tag;
  // @@protoc_insertion_point(parse_start:object_detection.Detections)
  for (;;) {
    ::std::pair< ::google::protobuf::uint32, bool> p = input->ReadTagWithCutoff(127);
    tag = p.first;
    if (!p.second) goto handle_unusual;
    switch (::google::protobuf::internal::WireFormatLite::GetTagFieldNumber(tag)) {
      // repeated .object_detection.Detections.Object objects = 1;
      case 1: {
        if (tag == 10) {
         parse_objects:
          DO_(::google::protobuf::internal::WireFormatLite::ReadMessageNoVirtual(
                input, add_objects()));
        } else {
          goto handle_unusual;
        }
        if (input->ExpectTag(10)) goto parse_objects;
        if (input->ExpectAtEnd()) goto success;
        break;
      }

      default: {
      handle_unusual:
        if (tag == 0 ||
            ::google::protobuf::internal::WireFormatLite::GetTagWireType(tag) ==
            ::google::protobuf::internal::WireFormatLite::WIRETYPE_END_GROUP) {
          goto success;
        }
        DO_(::google::protobuf::internal::WireFormat::SkipField(
              input, tag, mutable_unknown_fields()));
        break;
      }
    }
  }
success:
  // @@protoc_insertion_point(parse_success:object_detection.Detections)
  return true;
failure:
  // @@protoc_insertion_point(parse_failure:object_detection.Detections)
  return false;
#undef DO_
}

void Detections::SerializeWithCachedSizes(
    ::google::protobuf::io::CodedOutputStream* output) const {
  // @@protoc_insertion_point(serialize_start:object_detection.Detections)
  // repeated .object_detection.Detections.Object objects = 1;
  for (int i = 0; i < this->objects_size(); i++) {
    ::google::protobuf::internal::WireFormatLite::WriteMessageMaybeToArray(
      1, this->objects(i), output);
  }

  if (!unknown_fields().empty()) {
    ::google::protobuf::internal::WireFormat::SerializeUnknownFields(
        unknown_fields(), output);
  }
  // @@protoc_insertion_point(serialize_end:object_detection.Detections)
}

::google::protobuf::uint8* Detections::SerializeWithCachedSizesToArray(
    ::google::protobuf::uint8* target) const {
  // @@protoc_insertion_point(serialize_to_array_start:object_detection.Detections)
  // repeated .object_detection.Detections.Object objects = 1;
  for (int i = 0; i < this->objects_size(); i++) {
    target = ::google::protobuf::internal::WireFormatLite::
      WriteMessageNoVirtualToArray(
        1, this->objects(i), target);
  }

  if (!unknown_fields().empty()) {
    target = ::google::protobuf::internal::WireFormat::SerializeUnknownFieldsToArray(
        unknown_fields(), target);
  }
  // @@protoc_insertion_point(serialize_to_array_end:object_detection.Detections)
  return target;
}

int Detections::ByteSize() const {
  int total_size = 0;

  // repeated .object_detection.Detections.Object objects = 1;
  total_size += 1 * this->objects_size();
  for (int i = 0; i < this->objects_size(); i++) {
    total_size +=
      ::google::protobuf::internal::WireFormatLite::MessageSizeNoVirtual(
        this->objects(i));
  }

  if (!unknown_fields().empty()) {
    total_size +=
      ::google::protobuf::internal::WireFormat::ComputeUnknownFieldsSize(
        unknown_fields());
  }
  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
  _cached_size_ = total_size;
  GOOGLE_SAFE_CONCURRENT_WRITES_END();
  return total_size;
}

void Detections::MergeFrom(const ::google::protobuf::Message& from) {
  GOOGLE_CHECK_NE(&from, this);
  const Detections* source =
    ::google::protobuf::internal::dynamic_cast_if_available<const Detections*>(
      &from);
  if (source == NULL) {
    ::google::protobuf::internal::ReflectionOps::Merge(from, this);
  } else {
    MergeFrom(*source);
  }
}

void Detections::MergeFrom(const Detections& from) {
  GOOGLE_CHECK_NE(&from, this);
  objects_.MergeFrom(from.objects_);
  mutable_unknown_fields()->MergeFrom(from.unknown_fields());
}

void Detections::CopyFrom(const ::google::protobuf::Message& from) {
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void Detections::CopyFrom(const Detections& from) {
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool Detections::IsInitialized() const {

  if (!::google::protobuf::internal::AllAreInitialized(this->objects())) return false;
  return true;
}

void Detections::Swap(Detections* other) {
  if (other != this) {
    objects_.Swap(&other->objects_);
    std::swap(_has_bits_[0], other->_has_bits_[0]);
    _unknown_fields_.Swap(&other->_unknown_fields_);
    std::swap(_cached_size_, other->_cached_size_);
  }
}

::google::protobuf::Metadata Detections::GetMetadata() const {
  protobuf_AssignDescriptorsOnce();
  ::google::protobuf::Metadata metadata;
  metadata.descriptor = Detections_descriptor_;
  metadata.reflection = Detections_reflection_;
  return metadata;
}


// @@protoc_insertion_point(namespace_scope)

}  // namespace object_detection

// @@protoc_insertion_point(global_scope)
