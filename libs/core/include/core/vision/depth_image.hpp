#pragma once

#include <cstdint>
#include <vector>

namespace ddrl::core::vision {

struct DepthImage {
  std::uint16_t      width{0};
  std::uint16_t      height{0};
  std::vector<float> data;

  [[nodiscard]] std::size_t size() const noexcept { return data.size(); }
  void                      resize(std::uint16_t w, std::uint16_t h)
  {
    width  = w;
    height = h;
    data.resize(static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
  }
  [[nodiscard]] float& operator()(std::uint16_t row, std::uint16_t col)
  {
    return data[static_cast<std::size_t>(row) * width + col];
  }
  [[nodiscard]] const float& operator()(std::uint16_t row, std::uint16_t col) const
  {
    return data[static_cast<std::size_t>(row) * width + col];
  }
};

} // namespace ddrl::core::vision
