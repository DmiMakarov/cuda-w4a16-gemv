#ifndef QUANT_DEVICE_BUFFER_HPP
#define QUANT_DEVICE_BUFFER_HPP

#include <cuda_runtime.h>

#include <span>
#include <cstddef>
#include <format>
#include <stdexcept>
#include <utility>
#include <string_view>

namespace quant::gpu
{

namespace detail
{
    inline void check(cudaError_t err, std::string_view what)
    {
        if(err != cudaSuccess)
            throw std::runtime_error(std::format("{}: {}", what, cudaGetErrorString(err)));
    }
}

template <typename T>
class DeviceBuffer
{
public:
    explicit DeviceBuffer(std::size_t count)
    {
        cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&ptr_), count * sizeof(T));
        
        detail::check(err, std::format("Cant allocate memory in DeviceBuffer: try to allocate {} bytes", count * sizeof(T)));
        size_ = count;
    }
    ~DeviceBuffer()
    {
        cudaFree(ptr_);
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept 
        : ptr_(std::exchange(other.ptr_, nullptr)),
          size_(std::exchange(other.size_, 0)){}

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept
    {
        if(this != &other)
        {
            cudaFree(ptr_);
            this->ptr_ = std::exchange(other.ptr_, nullptr);
            this->size_ = std::exchange(other.size_, 0);
        }
        return *this;
    }

    void copy_from_host(std::span<const T> src)
    {
        if(src.size() != size_)
            throw std::invalid_argument(std::format("For copy to device the data size must be equal to than buffer size. Buffer size = {}, input size = {}", size_, src.size()));

        cudaError_t err = cudaMemcpy(ptr_, src.data(), size_ * sizeof(T), cudaMemcpyHostToDevice);
        detail::check(err, "Cant cudaMemcpu to device");
    }

    void copy_to_host(std::span<T> dst) const
    {
        if(dst.size() != size_)
            throw std::invalid_argument(std::format("For copy to host the data size must be equal to than buffer size. Buffer size = {}, input size = {}", size_, dst.size()));

        cudaError_t err = cudaMemcpy(dst.data(), ptr_, size_ * sizeof(T), cudaMemcpyDeviceToHost);
        detail::check(err, "Cant cudaMemcpu to host");
    }

    T* data() noexcept
    {
        return ptr_;
    }

    const T* data() const noexcept
    {
        return ptr_;
    }

    std::size_t size() const noexcept
    {
        return size_;
    }

private:
    T* ptr_ = nullptr;
    std::size_t size_ = 0;
};

} //namespace quant::gpu

#endif //QUANT_DEVICE_BUFFER_HPP