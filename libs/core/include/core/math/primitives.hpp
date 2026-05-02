#pragma once

#include <cmath>
#include <ostream>

namespace ddrl::core::math {

inline constexpr double kEps = 1e-9;

struct Vec2 {
  double x{0.0}, y{0.0};

  [[nodiscard]] constexpr Vec2 operator+(const Vec2 &o) const noexcept {
    return {.x=x + o.x, .y=y + o.y};
  }
  [[nodiscard]] constexpr Vec2 operator-(const Vec2 &o) const noexcept {
    return {.x=x - o.x, .y=y - o.y};
  }
  [[nodiscard]] constexpr Vec2 operator*(double s) const noexcept {
    return {.x=x * s, .y=y * s};
  }
  [[nodiscard]] constexpr Vec2 operator/(double s) const noexcept {
    return {.x=x / s , .y=y / s};
  }
  [[nodiscard]] constexpr Vec2 operator-() const noexcept {
    return {.x=-x, .y=-y};
  }

  Vec2 &operator+=(const Vec2 &o) noexcept {
    x += o.x;
    y += o.y;
    return *this;
  }
  Vec2 &operator-=(const Vec2 &o) noexcept {
    x -= o.x;
    y -= o.y;
    return *this;
  }
  Vec2 &operator*=(double s) noexcept {
    x *= s;
    y *= s;
    return *this;
  }
  Vec2 &operator/=(double s) noexcept {
    x /= s;
    y /= s;
    return *this;
  }

  [[nodiscard]] constexpr double dot(const Vec2 &o) const noexcept {
    return (x * o.x) + (y * o.y);
  }
  [[nodiscard]] constexpr double length_sq() const noexcept {
    return dot(*this);
  }
  [[nodiscard]] double length() const noexcept {
    return std::sqrt(length_sq());
  }

  [[nodiscard]] Vec2 normalized(double eps = kEps) const noexcept {
    const double len = length();
    return (len > eps) ? (*this / len) : Vec2{};
  }

  [[nodiscard]] double l2(const Vec2 &o) const noexcept {
    return std::hypot(x - o.x, y - o.y);
  }
};
constexpr Vec2 operator*(double s, const Vec2 &v) noexcept {
  return v * s;
}

struct Vec3 {
  double x{0.0}, y{0.0}, z{0.0};

  constexpr Vec3() = default;
  constexpr Vec3(double x_, double y_, double z_) noexcept
      : x(x_), y(y_), z(z_) {}

  [[nodiscard]] constexpr Vec3 operator+(const Vec3 &o) const noexcept {
    return {x + o.x, y + o.y, z + o.z};
  }
  [[nodiscard]] constexpr Vec3 operator-(const Vec3 &o) const noexcept {
    return {x - o.x, y - o.y, z - o.z};
  }
  [[nodiscard]] constexpr Vec3 operator-() const noexcept {
    return {-x, -y, -z};
  }
  [[nodiscard]] constexpr Vec3 operator*(double s) const noexcept {
    return {x * s, y * s, z * s};
  }
  [[nodiscard]] constexpr Vec3 operator/(double s) const noexcept {
    return {x / s, y / s, z / s};
  }

  Vec3 &operator+=(const Vec3 &o) noexcept {
    x += o.x;
    y += o.y;
    z += o.z;
    return *this;
  }
  Vec3 &operator-=(const Vec3 &o) noexcept {
    x -= o.x;
    y -= o.y;
    z -= o.z;
    return *this;
  }
  Vec3 &operator*=(double s) noexcept {
    x *= s;
    y *= s;
    z *= s;
    return *this;
  }
  Vec3 &operator/=(double s) noexcept {
    x /= s;
    y /= s;
    z /= s;
    return *this;
  }

  [[nodiscard]] constexpr double dot(const Vec3 &o) const noexcept {
    return (x * o.x) + (y * o.y) + (z * o.z);
  }
  [[nodiscard]] constexpr Vec3 cross(const Vec3 &o) const noexcept {
    return {(y * o.z) - (z * o.y), (z * o.x) - (x * o.z), (x * o.y) - (y * o.x)};
  }
  [[nodiscard]] constexpr double length_sq() const noexcept {
    return dot(*this);
  }
  [[nodiscard]] double length() const noexcept {
    return std::sqrt(length_sq());
  }

  [[nodiscard]] Vec3 normalized(double eps = kEps) const noexcept {
    const double len = length();
    return (len > eps) ? (*this / len) : Vec3{};
  }
};

struct Point2 {
  double x{0.0}, y{0.0};
  constexpr Point2() = default;
  constexpr Point2(double x_, double y_) noexcept : x(x_), y(y_) {}
};

// point ± vector -> point, point - point -> vector
[[nodiscard]] constexpr Point2 operator+(const Point2 &p,
                                         const Vec2 &v) noexcept {
  return {p.x + v.x, p.y + v.y};
}
[[nodiscard]] constexpr Point2 operator+(const Vec2 &v,
                                         const Point2 &p) noexcept {
  return p + v;
}
[[nodiscard]] constexpr Point2 operator-(const Point2 &p,
                                         const Vec2 &v) noexcept {
  return {p.x - v.x, p.y - v.y};
}
[[nodiscard]] constexpr Vec2 operator-(const Point2 &a,
                                       const Point2 &b) noexcept {
  return {a.x - b.x, a.y - b.y};
}

inline Point2 &operator+=(Point2 &p, const Vec2 &v) noexcept {
  p.x += v.x;
  p.y += v.y;
  return p;
}
inline Point2 &operator-=(Point2 &p, const Vec2 &v) noexcept {
  p.x -= v.x;
  p.y -= v.y;
  return p;
}

[[nodiscard]] inline double distance_sq(const Point2 &a,
                                        const Point2 &b) noexcept {
  const Vec2 d = a - b;
  return d.dot(d);
}
[[nodiscard]] inline double distance(const Point2 &a,
                                     const Point2 &b) noexcept {
  return std::sqrt(distance_sq(a, b));
}

struct Point3 {
  double x{0.0}, y{0.0}, z{0.0};
  constexpr Point3() = default;
  constexpr Point3(double x_, double y_, double z_) noexcept
      : x(x_), y(y_), z(z_) {}
};

[[nodiscard]] constexpr Point3 operator+(const Point3 &p,
                                         const Vec3 &v) noexcept {
  return {p.x + v.x, p.y + v.y, p.z + v.z};
}
[[nodiscard]] constexpr Point3 operator+(const Vec3 &v,
                                         const Point3 &p) noexcept {
  return p + v;
}
[[nodiscard]] constexpr Point3 operator-(const Point3 &p,
                                         const Vec3 &v) noexcept {
  return {p.x - v.x, p.y - v.y, p.z - v.z};
}
[[nodiscard]] constexpr Vec3 operator-(const Point3 &a,
                                       const Point3 &b) noexcept {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}

inline Point3 &operator+=(Point3 &p, const Vec3 &v) noexcept {
  p.x += v.x;
  p.y += v.y;
  p.z += v.z;
  return p;
}
inline Point3 &operator-=(Point3 &p, const Vec3 &v) noexcept {
  p.x -= v.x;
  p.y -= v.y;
  p.z -= v.z;
  return p;
}

struct Segment2 {
  Point2 p1, p2;
};

} // namespace ddrl::core::math



inline std::ostream &operator<<(std::ostream &os,
                                const ddrl::core::math::Point2 &p) {
  return os << "{" << p.x << ", " << p.y << "}";
}

inline std::ostream &operator<<(std::ostream &os,
                                const ddrl::core::math::Point3 &p) {
  return os << "{" << p.x << ", " << p.y << ", " << p.z << "}";
}

inline std::ostream &operator<<(std::ostream &os,
                                const ddrl::core::math::Vec2 &v) {
  return os << "{" << v.x << ", " << v.y << "}";
}

inline std::ostream &operator<<(std::ostream &os,
                                const ddrl::core::math::Vec3 &v) {
  return os << "{" << v.x << ", " << v.y << ", " << v.z << "}";
}

inline std::ostream &operator<<(std::ostream &os,
                                const ddrl::core::math::Segment2 &s) {
  return os << "Segment2(" << s.p1 << ", " << s.p2 << ")";
}
