#include "core/allocator.h"
#include <utility>

namespace infini {
Allocator::Allocator(Runtime runtime) : runtime(runtime) {
  used = 0;
  peak = 0;
  ptr = nullptr;

  // 'alignment' defaults to sizeof(uint64_t), because it is the length of
  // the longest data type currently supported by the DataType field of
  // the tensor
  alignment = sizeof(uint64_t);
}

Allocator::~Allocator() {
  if (this->ptr != nullptr) {
    runtime->dealloc(this->ptr);
  }
}

size_t Allocator::alloc(size_t size) {
  IT_ASSERT(this->ptr == nullptr);
  // pad the size to the multiple of alignment
  size = this->getAlignedSize(size);

  // =================================== 作业
  // ===================================
  // TODO: 设计一个算法来分配内存，返回起始地址偏移量
  // =================================== 作业
  // ===================================
  this->used += size;

  for (auto it = this->free_blocks.begin(); it != this->free_blocks.end();
       it++) {
    if (it->second >= size) {
      it->second -= size;
      size_t offset = it->first + it->second;
      if (!it->second)
        this->free_blocks.erase(it);
      return offset;
    }
  }

  this->peak += size;

  return this->peak - size;
}

void Allocator::free(size_t addr, size_t size) {
  IT_ASSERT(this->ptr == nullptr);
  size = getAlignedSize(size);

  // =================================== 作业
  // ===================================
  // TODO: 设计一个算法来回收内存
  // =================================== 作业
  // ===================================
  this->used -= size;
  if (this->peak - size == addr) {
    this->peak -= size;
    return;
  }

  auto it = this->free_blocks.lower_bound(addr);
  if (it != this->free_blocks.end()) {
    size_t l = it->first, r = it->first + it->second;
    if (addr + size == l) {
      if (it != this->free_blocks.begin()) {
        auto pre_it = it;
        pre_it--;
        size_t pre_l = it->first, pre_r = it->first + it->second;
        if (addr == pre_r) {
          this->free_blocks[pre_l] = pre_it->second + size + it->second;
          return;
        }
      } else {
        this->free_blocks[addr] = it->second + size;
        this->free_blocks.erase(it);
        return;
      }
    }
  }
  if (it != this->free_blocks.begin()) {
    it--;
    size_t l = it->first, r = it->first + it->second;
    if (addr == r) {
      this->free_blocks[l] = it->second + size;
      return;
    }
  }
  this->free_blocks[addr] = size;
}

void *Allocator::getPtr() {
  if (this->ptr == nullptr) {
    this->ptr = runtime->alloc(this->peak);
    printf("Allocator really alloc: %p %lu bytes\n", this->ptr, peak);
  }
  return this->ptr;
}

size_t Allocator::getAlignedSize(size_t size) {
  return ((size - 1) / this->alignment + 1) * this->alignment;
}

void Allocator::info() {
  std::cout << "Used memory: " << this->used << ", peak memory: " << this->peak
            << std::endl;
}
} // namespace infini
