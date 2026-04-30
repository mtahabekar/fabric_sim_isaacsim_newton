std_msgs/Header header

bool enabled
bool active
bool valid_distance
# True when the top CBF input velocity topics are fresh.
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

float64[] jacobian
# QP nominal input: top-left/top-right velocities mapped as desired
# bottom-left/bottom-right velocities.
float64[] nominal_velocity
float64[] safe_velocity
float64[] correction

geometry_msgs/Vector3 normal
geometry_msgs/Point closest_particle
geometry_msgs/Point closest_obstacle_point
