// Copyright 2023 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "pir/dense_dpf_pir_database.h"

#include <stddef.h>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/numeric/int128.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "dpf/status_macros.h"
#include "pir/internal/inner_product_hwy.h"

namespace distributed_point_functions {

namespace {

// Returns the number of bytes occupied by a value of `value_size_in_bytes`
// when aligned according to the block type.
size_t AlignBytes(size_t value_size_in_bytes) {
  constexpr size_t kAlignmentSize =
      sizeof(typename DenseDpfPirDatabase::BlockType);
  constexpr size_t kAlignmentMask = ~(kAlignmentSize - 1);
  // The number of aligned bytes is the least multiple of kAlignmentSize larger
  // than `value_size_in_bytes`, i.e. it is
  //   ceil(value_size_in_bytes / kAlignmentSize) * kAlignmentSize.
  // The division and subsequent multiplication is saved by simply masking off
  // the lowest `kAlignmentSize` bits.
  size_t round_up_value_size_in_bytes =
      (value_size_in_bytes + kAlignmentSize - 1);
  return round_up_value_size_in_bytes & kAlignmentMask;
}


inline constexpr int64_t NumBytesToNumBlocks(int64_t num_bytes) {
  return (num_bytes + (sizeof(DenseDpfPirDatabase::BlockType) - 1)) /
         sizeof(DenseDpfPirDatabase::BlockType);
}

}  // namespace

DenseDpfPirDatabase::Builder::Builder()
    : total_database_bytes_(0) {}

std::unique_ptr<DenseDpfPirDatabase::Interface::Builder>
DenseDpfPirDatabase::Builder::Clone() const {
  auto result = std::make_unique<Builder>();
  result->total_database_bytes_ = total_database_bytes_;
  result->values_ = values_;
  return result;
}

DenseDpfPirDatabase::Builder& DenseDpfPirDatabase::Builder::Insert(
    std::string value) {
  total_database_bytes_ += AlignBytes(value.size());
  values_.push_back(std::move(value));
  return *this;
}

DenseDpfPirDatabase::Builder& DenseDpfPirDatabase::Builder::Clear() {
  values_.clear();
  total_database_bytes_ = 0;
  return *this;
}

absl::StatusOr<std::unique_ptr<DenseDpfPirDatabase::Interface>>
DenseDpfPirDatabase::Builder::Build() {
  auto database = absl::WrapUnique(
      new DenseDpfPirDatabase(values_.size(), total_database_bytes_));
  std::vector<std::string> values =
      std::move(values_);  // Ensures values are freed after returning.
  for (std::string& value : values) {
    DPF_RETURN_IF_ERROR(database->Append(std::move(value)));
  }
  return database;
}

DenseDpfPirDatabase::DenseDpfPirDatabase(int64_t num_values,
                                         int64_t total_database_bytes)
    : max_value_size_(0) {
  // Reserve space for storing the desired number of bytes
  buffer_.reserve(NumBytesToNumBlocks(total_database_bytes));
  // Reserve space for storing the database values
  value_offsets_.reserve(num_values);
  content_views_.reserve(num_values);
}

// Appends a record `value` at the current end of the database.
absl::Status DenseDpfPirDatabase::Append(std::string value) {
  // The new value will be stored at the end of the current buffer space.
  const size_t offset = buffer_.size();
  const size_t value_size = value.size();
  if (value_size == 0) {
    // We have an empty value, so we store its offset and return.
    value_offsets_.push_back({offset, 0});
    content_views_.push_back(absl::string_view());
    return absl::OkStatus();
  }

  // Number of buffer elements needed to store the aligned value
  const size_t value_size_aligned = AlignBytes(value_size);
  const size_t num_additional_blocks = value_size_aligned / sizeof(BlockType);
  const size_t num_existing_blocks = buffer_.capacity();
  // Save the old buffer head pointer to ensure it is not reallocated, which
  // would invalidate existing content_views.
  const BlockType* const buffer_head_old = buffer_.data();
  if (offset + num_additional_blocks > num_existing_blocks) {
    // We don't have enough space in the buffer for this element. This signals
    // an implementation error in Buider::Build().
    return absl::InternalError(
        "Not enough buffer space available. This should not happen.");
  }
  buffer_.resize(buffer_.size() + num_additional_blocks);
  if (buffer_head_old != &buffer_.at(0)) {
    return absl::InternalError(
        "Buffer was reallocated unexpectedly. This should not happen.");
  }

  // Append the value to the buffer
  char* const buffer_at_offset = reinterpret_cast<char*>(&buffer_.at(offset));
  value.copy(buffer_at_offset, value_size);
  if (value_size > max_value_size_) {
    max_value_size_ = value_size;
  }

  // Store the position and the view of the value in `buffer_`.
  value_offsets_.push_back({offset, value_size});
  content_views_.push_back(absl::string_view(buffer_at_offset, value_size));
  return absl::OkStatus();
}

absl::Status DenseDpfPirDatabase::UpdateEntry(size_t index, std::string new_value) {
  return BatchUpdateEntry({index}, {new_value});
}

absl::Status DenseDpfPirDatabase::BatchUpdateEntry(
    const std::vector<size_t>& indices,
    const std::vector<std::string>& new_values) {
  if (indices.size() != new_values.size()) {
    return absl::InvalidArgumentError("Indices and values size mismatch");
  }
  
  std::vector<bool> will_update(size(), false);
  
  size_t additional_bytes_needed = 0;
  size_t current_offset = buffer_.size();
  
  for (size_t i = 0; i < indices.size(); ++i) {
    const size_t index = indices[i];
    if (index >= size()) {
      return absl::OutOfRangeError("Index out of bounds");
    }
    
    will_update[index] = true;
    
    const auto& [current_offset_i, current_size] = value_offsets_[index];
    const size_t new_size = new_values[i].size();
    const size_t current_aligned_size = AlignBytes(current_size);
    const size_t new_aligned_size = AlignBytes(new_size);
    
    if (new_aligned_size > current_aligned_size) {
      additional_bytes_needed += new_aligned_size;
    }
  }
  
  const BlockType* const buffer_head_old = buffer_.data();
  
  if (additional_bytes_needed > 0) {
    const size_t new_blocks_needed = (additional_bytes_needed + sizeof(BlockType) - 1) / sizeof(BlockType);
    buffer_.reserve(buffer_.size() + new_blocks_needed);
    buffer_.resize(buffer_.size() + new_blocks_needed);
  }
  
  const bool reallocation_occurred = (buffer_head_old != buffer_.data() && buffer_head_old != nullptr);
  if (reallocation_occurred) {
    for (size_t i = 0; i < value_offsets_.size(); ++i) {
      if (!will_update[i]) {
        const auto& [offset_i, size_i] = value_offsets_[i];
        char* const rebased_buffer = reinterpret_cast<char*>(&buffer_[offset_i]);
        content_views_[i] = absl::string_view(rebased_buffer, size_i);
      }
    }
  }
  
  // Perform updates
  for (size_t i = 0; i < indices.size(); ++i) {
    const size_t index = indices[i];
    const std::string& new_value = new_values[i];
    const size_t new_size = new_value.size();
    const auto& [current_offset_i, current_size] = value_offsets_[index];
    const size_t current_aligned_size = AlignBytes(current_size);
    const size_t new_aligned_size = AlignBytes(new_size);
    
    if (new_aligned_size <= current_aligned_size) {
      // Update in-place using memcpy instead of copy for better performance
      char* const buffer_at_offset = reinterpret_cast<char*>(&buffer_[current_offset_i]);
      std::memset(buffer_at_offset, 0, current_aligned_size);
      if (new_size > 0) {
        std::memcpy(buffer_at_offset, new_value.data(), new_size);
      }
      value_offsets_[index].second = new_size;
      content_views_[index] = absl::string_view(buffer_at_offset, new_size);
    } else {
      // Append to end using memcpy
      char* const buffer_at_offset = reinterpret_cast<char*>(&buffer_[current_offset]);
      std::memcpy(buffer_at_offset, new_value.data(), new_size);
      value_offsets_[index] = {current_offset, new_size};
      content_views_[index] = absl::string_view(buffer_at_offset, new_size);
      current_offset += new_aligned_size / sizeof(BlockType);
    }
    
    max_value_size_ = std::max(max_value_size_, new_size);
  }
  
  return absl::OkStatus();
}

// Returns the inner product between the database values and a bit vector
// (packed in blocks).
absl::StatusOr<std::vector<std::string>> DenseDpfPirDatabase::InnerProductWith(
    absl::Span<const std::vector<BlockType>> selections) const {
  return pir_internal::InnerProduct(content_views_, selections,
                                    max_value_size_);
}

}  // namespace distributed_point_functions
