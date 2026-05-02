
#include <type_traits>
namespace ddrl::core {

template <typename E>
E next_enum(E e, E count)
{
  using U = std::underlying_type_t<E>;
  return static_cast<E>((static_cast<U>(e) + 1) % static_cast<U>(count));
}

} // namespace ddrl::core
