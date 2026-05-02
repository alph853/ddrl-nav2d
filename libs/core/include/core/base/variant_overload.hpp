#pragma once

namespace ddrl::core {
// Helper for std::visit with multiple variants
template <class... Ts> struct Overloaded : Ts... {
  using Ts::operator()...;
};
template <class... Ts> Overloaded(Ts...) -> Overloaded<Ts...>;

} // namespace ddrl::core
