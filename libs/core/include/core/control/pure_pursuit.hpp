#pragma once

namespace ddrl::core::control {

struct PurePursuitCaps {
  double lookahead_m;
  double min_lookahead_m;
  double max_lookahead_m;
  bool   speed_coupled_to_curvature; // slow down on tight turns
  double kv; // optional gain to tie speed to curvature (v = v0/(1+kv*|kappa|))
};

} // namespace ddrl::core::control
