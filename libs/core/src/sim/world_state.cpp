#include "core/sim/world_state.hpp"

std::string to_string(const ddrl::core::sim::RobotState& state)
{
  return std::format(
    "RobotState(model_id='{}', pose=({}, {}, {}), velocity=({}, {}, {}), steering_angle={}, "
    "speed={}, "
    "is_collided={})",
    state.model_id,
    state.pose.pos.x,
    state.pose.pos.y,
    state.pose.yaw,
    state.velocity.linear.x,
    state.velocity.linear.y,
    state.velocity.angular,
    state.steering_angle,
    state.speed,
    state.is_collided
  );
}

std::string to_string(const ddrl::core::sim::ObjectState& state)
{
  return std::format(
    "ObjectState(id={}, name='{}', model_id='{}', pose=({}, {}, {}), velocity=({}, {}, {}), "
    "state_name='{}')",
    state.id,
    state.name,
    state.model_id,
    state.pose.pos.x,
    state.pose.pos.y,
    state.pose.yaw,
    state.velocity.linear.x,
    state.velocity.linear.y,
    state.velocity.angular,
    state.state_name
  );
}

std::string to_string(std::span<const ddrl::core::sim::ObjectState> states)
{
  if (states.empty()) {
    return "ObjectState[]";
  }

  std::string result = "ObjectState[\n";
  for (const auto& state : states) {
    result += "  " + to_string(state) + "\n";
  }
  result += "]";
  return result;
}

std::string to_string(const ddrl::core::sim::WorldStateSnapshot& snapshot)
{
  return std::format(
    "WorldStateSnapshot(sim_time={}, robot={}, route='{}' waypoint={}/{}, dynamic_objects=[...], sensor_clouds_count={})",
    snapshot.sim_time,
    to_string(snapshot.robot),
    snapshot.active_route_name,
    snapshot.active_waypoint_index,
    snapshot.active_waypoint_count,
    to_string(snapshot.dynamic_objects),
    snapshot.sensor_clouds.size()
  );
}

std::string to_string(std::span<const ddrl::core::sim::WorldStateSnapshot> snapshot_ptr)
{
  if (snapshot_ptr.empty()) {
    return "WorldStateSnapshot[]";
  }

  std::string result = "WorldStateSnapshot[\n";
  for (const auto& snapshot : snapshot_ptr) {
    result += "  " + to_string(snapshot) + "\n";
  }
  result += "]";
  return result;
}
