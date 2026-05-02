#include <sstream>
#include <string_view>
#include <vector>

namespace ddrl::core {

inline auto split(std::string_view str, char delimiter)
{
  std::vector<std::string_view> result;

  size_t start = 0;
  while (true) {
    size_t pos = str.find(delimiter, start);
    if (pos == std::string_view::npos) {
      result.emplace_back(str.substr(start));
      return result;
    }
    result.emplace_back(str.substr(start, pos - start));
    start = pos + 1;
  }
}

inline auto join(const std::vector<std::string>& parts, std::string_view delimiter)
{
  if (parts.empty()) {
    return std::string{};
  }
  std::ostringstream oss;
  for (size_t i = 0; i < parts.size() - 1; ++i) {
    oss << parts[i] << delimiter;
  }
  oss << parts.back();
  return oss.str();
}

} // namespace ddrl::core
