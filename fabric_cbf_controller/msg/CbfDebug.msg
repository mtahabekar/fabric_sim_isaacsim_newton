std_msgs/Header header

bool enabled
bool active
bool valid_distance
# True when the leader CBF input velocity topics are fresh.
bool nominal_ready
bool command_published

string mode
string solver_status
string obstacle_id

float64 h
float64 min_distance
float64 safety_margin
float64 alpha_h
float64 lhs_nominal
float64 lhs_safe
float64 rhs
float64 compute_time_ms

float64[] jacobian
# QP nominal input: leader-left/leader-right velocities mapped as desired
# follower-left/follower-right velocities. Legacy top/bottom topic aliases may
# still be enabled for backward compatibility.
float64[] nominal_velocity
float64[] safe_velocity
float64[] correction

geometry_msgs/Vector3 normal
geometry_msgs/Point closest_particle
geometry_msgs/Point closest_obstacle_point

# Stretch avoidance barriers (one per active pair)
string[] stretch_pair_names
float64[] stretch_h_values
float64[] stretch_distances
float64[] stretch_d_max_values
bool stretch_active
