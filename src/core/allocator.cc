#include "core/allocator.h"
#include <utility>
#include <algorithm> // for std::max


namespace infini
{
    Allocator::Allocator(Runtime runtime) : runtime(runtime)
    {
        used = 0;
        peak = 0;
        ptr = nullptr;

        // 'alignment' defaults to sizeof(uint64_t), because it is the length of
        // the longest data type currently supported by the DataType field of
        // the tensor
        alignment = sizeof(uint64_t);
    }

    Allocator::~Allocator()
    {
        if (this->ptr != nullptr)
        {
            runtime->dealloc(this->ptr);
        }
    }

    size_t Allocator::alloc(size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        // pad the size to the multiple of alignment
        size = this->getAlignedSize(size);

         // =================================== 作业 ===================================
        // TODO: 设计一个算法来分配内存，返回起始地址偏移量
        // =================================== 作业 ===================================
        
        //循环遍历map，找到第一个比size大的内存块
        for (auto it = freeBlocks.begin(); it != freeBlocks.end(); ++it) {
            if (it->second >= size) {
                // Found a suitable block
                size_t addr = it->first;
                size_t remaining = it->second - size;
                
                // Remove the block from free list
                freeBlocks.erase(it);
                
                // If there's remaining space, add it back to free list
                if (remaining > 0) {
                    freeBlocks[addr + size] = remaining;
                }
                
                used += size;
                peak = std::max(peak, used);
                return addr;
            }
        }
        
        // No suitable free block found, allocate new memory
        size_t addr = peak;
        peak += size;
        used += size;
        return addr;
    
    }

    void Allocator::free(size_t addr, size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        size = getAlignedSize(size);

        // Add the freed block to free list
        // Update memory usage
        // used -= size;

        // Try to merge the freed block with neighboring blocks
        size_t start = addr;
        size_t end = addr + size;
         // 如果释放的内存块正好是最后一个内存块
        if (addr + size == this->peak)
        {
            this->peak -= size;
            return;
        }
        // 找到第一个不小于addr的块，如何插入，应该是后一块
        auto next = freeBlocks.lower_bound(addr);

        // 检查合并前一块
        if (next != freeBlocks.begin()) {
            auto prev = std::prev(next);
            if (prev->first + prev->second == addr) {
                // Merge with previous block
                start = prev->first;
                size += prev->second;
                freeBlocks.erase(prev);
            }
        }

        // 检查合并后一块
        if (next != freeBlocks.end() && next->first == end) {
            // Merge with next block
            size += next->second;
            freeBlocks.erase(next);
        }

        // 插入空闲快
        freeBlocks[start] = size;
        used -= size;

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


