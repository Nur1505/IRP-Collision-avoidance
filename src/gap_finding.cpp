#include <ros/ros.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/TwistStamped.h>
#include <hector_uav_msgs/EnableMotors.h>
#include <std_srvs/Empty.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

// ============================================================================
// Online Lucas-Kanade optical flow + Visual Potential Field (VPF) +
// Artificial Potential Field (APF) avoidance.
//
//   - LK + VPF + APF pipeline.
//   - LATERAL collision avoidance (vx forward, vy lateral) with a small
//     heading-recovery controller that regulates yaw back toward zero.
//   - Measured yaw-rate optical flow is compensated before TTC and VPF
//     calculations so heading recovery is not mistaken for obstacle looming.
//   - The first controlled VPF test uses geometry-only row filtering.
//   - The noisy per-frame VPF repulsion signal is smoothed once, at its
//     source (right after it's computed from the vision pipeline), rather
//     than smoothing the final mixed command downstream. That earlier
//     approach (smoothing the combined command) doesn't fix upstream noise -
//     it just adds latency everywhere, including to genuine avoidance need.
//   - Optional rotational APF component. The visual VPF is condensed into a
//     dominant obstacle bearing; the radial obstacle-away direction is rotated
//     by +/-90 degrees to create a true tangential X-Y correction. The rotation
//     direction is selected from the planned-path/obstacle bearing difference
//     and held until the obstacle has been bypassed.
//   - The same executable can run the baseline controller by setting
//       _enable_rotational_apf:=false
// ============================================================================

class OnlineLKAPFVelocity
{
public:
    OnlineLKAPFVelocity()
        : nh_("~"),
          prev_time_(0.0),
          have_prev_image_(false),
          have_pose_(false),
          have_ttc_filtered_(false),
          have_prev_distance_to_goal_(false),
          have_progress_timer_start_(false),
          have_last_oscillation_time_(false),
          have_oscillation_episode_start_(false)
    {
        last_vx_ = 0.0;
        last_vy_ = 0.0;
        last_vz_ = 0.0;

        // ------------------------------------------------------
        // Path-following / corridor-keeping (attraction side of APF)
        // ------------------------------------------------------
        // This keeps the vehicle near the straight start->goal line so that
        // after an avoidance manoeuvre it's pulled back onto route instead of
        // drifting away in y indefinitely.
        use_path_controller_ = true;
        k_path_y_ = 0.6;
        k_goal_y_ = 0;   // small direct pull toward the final goal_y

        // Instead of pulling the UAV back to the closest point on the
        // start-to-goal line, follow a point a fixed distance ahead of the
        // current projection onto that line. This produces a gradual route
        // recovery after avoidance rather than an aggressive sideways return.
        nh_.param("path_lookahead_distance_m",
                  path_lookahead_distance_,
                  5.0);
        nh_.param("max_path_recovery_speed_mps",
                  max_path_recovery_speed_,
                  1.0);

        last_y_ref_ = 0.0;
        last_cross_track_error_ = 0.0;
        last_v_path_y_ = 0.0;
        last_v_goal_y_ = 0.0;

        // Yaw state is used by velocity-heading control and by measured
        // yaw-flow compensation before TTC and VPF calculations.
        current_yaw_ = 0.0;
        current_yaw_rate_ = 0.0;
        yaw_rate_filter_alpha_ = 0.25;

        k_yaw_recover_ = 0.6;
        max_yaw_recover_rate_ = 0.3;

        // ------------------------------------------------------
        // Velocity-heading control and dynamic translational FOE.
        // ------------------------------------------------------
        // The forward camera is yawed toward the requested horizontal motion
        // direction. While the camera is still turning, translational motion is
        // reduced and body-lateral crabbing is limited. TTC/VPF no longer assume
        // that the FOE is fixed at the image centre: the FOE is recomputed from
        // measured body-frame translation on every usable image.
      nh_.param("enable_velocity_heading_control",
                  enable_velocity_heading_control_,
                  true);
        nh_.param("k_velocity_heading",
                  k_velocity_heading_,
                  0.8);
        nh_.param("max_velocity_heading_rate",
                  max_velocity_heading_rate_,
                  0.5);
        nh_.param("heading_min_horizontal_speed",
                  heading_min_horizontal_speed_,
                  0.20);
        nh_.param("heading_slow_angle_deg",
                  heading_slow_angle_deg_,
                  25.0);
        nh_.param("heading_strong_slow_angle_deg",
                  heading_strong_slow_angle_deg_,
                  45.0);
        nh_.param("heading_slow_scale",
                  heading_slow_scale_,
                  0.60);
        nh_.param("heading_strong_slow_scale",
                  heading_strong_slow_scale_,
                  0.30);
        nh_.param("max_body_lateral_speed",
                  max_body_lateral_speed_,
                  0.60);

        // Legacy centre-FOE parameters are still read so existing launch/rosrun
        // commands remain compatible.  The restrictive lateral-ratio and heading-
        // error gates are no longer used for fresh perception.  Instead, the FOE
        // is computed directly from measured body-frame translation each frame.
        nh_.param("enable_centre_foe_validity_gate",
                  enable_centre_foe_validity_gate_,
                  true);
        nh_.param("centre_foe_min_forward_body_speed",
                  centre_foe_min_forward_body_speed_,
                  0.40);
        nh_.param("centre_foe_max_lateral_forward_ratio",
                  centre_foe_max_lateral_forward_ratio_,
                  0.35);
        nh_.param("centre_foe_max_heading_error_deg",
                  centre_foe_max_heading_error_deg_,
                  20.0);
        nh_.param("centre_foe_ttc_hold_s",
                  centre_foe_ttc_hold_s_,
                  1.00);
        nh_.param("centre_foe_risk_decay",
                  centre_foe_risk_decay_,
                  0.985);

        // Dynamic translational FOE.  For camera optical axis body +x,
        // image-right = body -y and image-down = body -z:
        //   u_FOE = cx - fx * vy_body / vx_body
        //   v_FOE = cy - fy * vz_body / vx_body
        // Only a small positive forward component is required.  This keeps TTC
        // and VPF alive while the UAV is yawing/crabbing instead of rejecting the
        // entire image whenever heading error or lateral velocity becomes large.
        nh_.param("dynamic_foe_min_forward_body_speed",
                  dynamic_foe_min_forward_body_speed_,
                  0.25);
        nh_.param("dynamic_foe_max_offset_image_widths",
                  dynamic_foe_max_offset_image_widths_,
                  8.0);

        current_body_vx_ = 0.0;
        current_body_vy_ = 0.0;
        current_body_vz_ = 0.0;
        desired_heading_ = 0.0;
        last_heading_error_ = 0.0;
        last_motion_alignment_scale_ = 1.0;
        centre_foe_model_valid_ = false;  // legacy CSV name; now means dynamic FOE valid
        last_centre_foe_valid_time_ = 0.0;
        have_last_centre_foe_valid_time_ = false;
        dynamic_foe_x_px_ = 320.0;
        dynamic_foe_y_px_ = 240.0;
        dynamic_foe_valid_ = false;
        frontal_corridor_x_min_px_ = 0.0;
        frontal_corridor_x_max_px_ = 0.0;

        // Camera model for yaw-flow compensation. The defaults match a
        // 640-pixel-wide camera with approximately 80 degree horizontal FOV.
        // They can be overridden from rosrun without editing the source.
        nh_.param("enable_yaw_flow_compensation",
                  enable_yaw_flow_compensation_,
                  true);
        nh_.param("camera_fx_pixels",
                  camera_fx_pixels_,
                  381.36);
        nh_.param("camera_fy_pixels",
                  camera_fy_pixels_,
                  381.36);
        nh_.param("yaw_flow_sign",
                  yaw_flow_sign_,
                  1.0);
        nh_.param("max_yaw_rate_for_compensation",
                  max_yaw_rate_for_compensation_,
                  1.5);

        goal_tolerance_ = 0.5;
        reached_goal_ = false;
        takeoff_done_ = false;

        collision_count_ = 0;
        in_collision_ = false;
        results_printed_ = false;

        // Safety metrics based on surface-to-surface obstacle clearance.
        // uav_collision_radius_ matches the Hector quadrotor collision mesh;
        // obstacle_radius_ matches the pole collision cylinders in the world.
        uav_collision_radius_ = 0.40;
        obstacle_radius_ = 0.30;
        current_obstacle_clearance_ = std::numeric_limits<double>::max();
        minimum_obstacle_clearance_ = std::numeric_limits<double>::max();
        minimum_collision_free_clearance_ = std::numeric_limits<double>::max();
        minimum_clearance_obstacle_x_ = 0.0;
        minimum_clearance_obstacle_y_ = 0.0;
        minimum_safe_clearance_obstacle_x_ = 0.0;
        minimum_safe_clearance_obstacle_y_ = 0.0;

        // Mission result metrics
        path_length_ = 0.0;
        mission_start_time_ = 0.0;
        mission_end_time_ = 0.0;
        mission_timer_started_ = false;
        have_last_path_pose_ = false;
        last_path_x_ = 0.0;
        last_path_y_ = 0.0;
        last_path_z_ = 0.0;

        path_files_opened_ = false;
        results_file_path_ = "/root/catkin_ws/experiment_results.csv";

        // Repeated-run configuration, overridable from rosrun:
        //   _target_runs:=N  _no_movement_timeout_s:=S  _movement_threshold_m:=M
        //   _reset_delay_s:=S  _stabilization_time_s:=S  _gazebo_reset_service:=srv
        nh_.param("target_runs", target_runs_, 1);
        nh_.param("no_movement_timeout_s", no_movement_timeout_s_, 120.0);
        nh_.param("movement_threshold_m", movement_threshold_m_, 0.25);
        nh_.param("reset_delay_s", reset_delay_s_, 2.0);
        nh_.param("stabilization_time_s", stabilization_time_s_, 3.0);
        nh_.param<std::string>("gazebo_reset_service",
                               gazebo_reset_service_,
                               "/gazebo/reset_world");

        current_run_ = 1;
        completed_runs_ = 0;
        successful_runs_ = 0;
        attempt_finished_ = false;
        resetting_world_ = false;
        stabilizing_after_reset_ = false;
        stabilization_end_time_ = 0.0;
        experiment_complete_ = false;

        movement_watch_started_ = false;
        last_movement_time_ = 0.0;
        movement_anchor_x_ = 0.0;
        movement_anchor_y_ = 0.0;
        //{10.5, 10.8}

        obstacles_medium_forest = {
            {-12.0, 5.5}, {-11.0, 8.0}, {-9.5, 4.0}, {1, 2}, {-7.0, 6.5},
            {-6.0, 10.5}, {-4.5, 7.5}, {-3.5, 11.0}, {-2.0, 4.5}, {-0.5, 12.0},
            {1.0, 9.0}, {2.5, 11.5}, {4.0, 8.0}, {5.5, 10.5}, {7.5, 7.5},
            {8.0, 11.5}, {9.5, 8.5}, {12.5, 11.8}, {14.5, 8.5},
            {-8.0, -1.0}, {3.0, 4.0}, {8.0, 3.0}, {13.0, 4.5},
            {-13.5, -5.0}, {-10.5, -9.0}, {-7.5, -5.5}, {-5.0, -12.0},
            {-2.5, -4.0}, {0.0, -8.5}, {2.5, -13.0}, {4.5, -3.0},
            {6.0, -9.5}, {8.5, -5.5}, {10.0, -12.0}, {11.5, -2.0},
            {13.5, -7.5}, {15.5, -4.0}, {17.0, -10.5}, {18.5, -1.5}
        };
std::vector<std::pair<double, double>> obstacles_medium_forest_0p10 = {
    {-9.277, -11.742},
    {-5.645, -10.983},
    {-2.306, -11.730},
    {0.002, -11.852},
    {3.442, -11.189},
    {7.255, -11.945},
    {10.081, -11.093},
    {12.876, -11.918},
    {17.025, -11.741},
    // {-12.555, -7.954},
    {-9.198, -8.382},
    {-5.886, -7.945},
    {-2.958, -8.251},
    {0.007, -8.538},
    {3.552, -8.833},
    {7.213, -8.703},
    {9.471, -8.767},
    {13.300, -8.071},
    {16.085, -8.621},
    // {-12.159, -4.729},
    {-9.846, -5.409},
    {-5.979, -5.612},
    {-2.273, -5.770},
    {0.889, -4.852},
    {3.314, -5.348},
    {6.464, -5.709},
    {10.371, -4.670},
    {13.367, -4.601},
    {16.189, -4.676},
    // {-11.916, -2.660},
    {-9.236, -2.137},
    {-6.454, -2.145},
    {-2.752, -2.321},
    {-0.079, -2.293},
    {3.964, -1.652},
    {6.574, -2.178},
    {10.522, -2.219},
    {13.273, -1.601},
    {16.920, -1.481},
    // {-12.304, 1.201},
    {-9.781, 0.757},
    {-5.444, 1.160},
    {-2.396, 0.979},
    {-0.008, 0.730},
    {3.015, 0.851},
    {6.518, 1.081},
    {9.928, 1.042},
    {13.233, 0.484},
    {15.825, 0.553},
    // {-12.309, 4.106},
    {-9.288, 4.303},
    {-5.726, 4.762},
    {-2.368, 3.671},
    {-0.019, 3.589},
    {3.973, 4.632},
    {6.302, 4.150},
    {9.417, 3.806},
    {13.108, 3.910},
    {16.786, 3.719},
    // {-12.084, 6.941},
    {-8.850, 7.766},
    {-6.488, 6.837},
    {-3.203, 7.223},
    {0.504, 7.310},
    {3.327, 6.811},
    {6.194, 7.454},
    {9.499, 7.073},
    {12.703, 7.604},
    {16.340, 7.774},
    // {-12.636, 10.852},
    {-8.598, 9.951},
    {-5.780, 10.752},
    {-2.756, 10.766},
    {0.213, 10.157},
    {3.278, 10.877},
    {6.748, 10.834},
    {10.431, 10.898},
    {13.101, 10.624},
    {15.829, 10.329}
};

std::vector<std::pair<double, double>> obstacles_vpf_gap_finding = {
    {6.378, -12.090},
    {-5.064, -7.254},
    {9.425, 3.811},
    {14.314, -10.579},
    {-0.452, -11.973},
    {-6.835, -0.369},
    {-12.867, -7.848},
    {6.706, 0.597},
    {11.716, -12.541},
    {11.603, 4.335},
    {-3.016, -8.906},
    {16.356, -4.487},
    {-10.788, -10.340},
    {12.911, 2.031},
    {3.138, 11.044},
    {-1.814, 0.770},
    {-6.544, -5.639},
    {-10.529, -5.917},
    {6.260, -3.798},
    {-5.317, 10.154},
    {-8.326, 5.091},
    {-8.569, -3.441},
    {17.371, 2.916},
    {3.788, 4.005},
    {12.766, 6.234},
    {-6.508, -11.917},
    {13.818, -5.022},
    {15.017, -1.504},
    {3.927, -6.289},
    {-1.159, -7.349},
    {17.623, -0.268},
    {11.171, -2.399},
    {-11.705, -3.388},
    {16.792, 8.303},
    {-13.339, 4.886},
    {0.547, 10.573},
    {2.018, -8.341},
    {5.422, -8.971},
    {10.243, 0.461},
    {-13.682, -4.791},
    {-13.088, 9.970},
    {-4.044, -11.287},
    {13.870, 10.406},
    {10.347, -9.567},
    {1.224, 0.715},
    {-6.973, -9.472},
    {15.675, 1.233},
    {-0.400, -1.305},
    {17.203, -10.299},
    {-1.058, -4.421},
    {3.580, -11.466},
    {12.949, -8.642},
    {-1.800, 11.342},
    {-5.373, 6.431},
    {16.360, 11.588},
    {-8.757, 10.743},
    {-6.205, -3.043},
    {5.507, 11.388},
    {8.833, -7.732},
    {6.215, -6.259},
    {11.298, 8.301},
    {11.758, -6.190},
    {13.686, 8.250},
    {-3.393, 8.390},
    {9.176, -5.046},
    {-3.988, -2.735},
    {-4.250, 0.670},
    {9.880, 6.089},
    {8.482, -11.133},
    {6.765, 5.703},
    {16.118, -7.836},
    {-10.500, 6.149},
    {2.187, -4.374},
    {8.509, -2.074},
    {3.199, -0.903},
    {-13.060, -0.250},
    {-8.740, 0},
    {-4.420, 3.750},
    {-0.100, 5.750},
    {4.220, 7.750},
};
        obstacles_zigzag = {
            {2.0, 3.0}, {4.0, 7.0}, {6.0, 3.0}, {8.0, 7.0}, {10.0, 5.0}
        };

        obstacles_u_shape_trap = {
            {2.0, 2.0}, {4.0, 2.0}, {6.0, 2.0}, {8.0, 2.0}, {10.0, 2.0}, {12.0, 2.0},
            {2.0, 10.0}, {4.0, 10.0}, {6.0, 10.0}, {8.0, 10.0}, {10.0, 10.0}, {12.0, 10.0},
            {12.0, 3.3}, {12.3, 4.6}, {12.0, 5.9}, {12.3, 7.2}, {12.0, 8.5}, {12.3, 9.3},
            
        };
//{6.5, 5.3}
        std::vector<std::pair<double, double>> obstacles_sparse_forest = {
            {-12.0, 2.5},
            {-4.0, 4.8},
            {4.0, 7.0},
            {-15.0, 10.0},
            {-10.0, -4.0},
            {-7.0, 11.5},
            {-1.0, -3.0},
            {0.0, 12.5},
            {8.0, 13.0},
            {12.0, 3.5},
            {15.0, 14.0},
            {18.0, -2.0}
        };

        std::vector<std::pair<double, double>> obstacles_sparse_alternating = {
            {-12.0, 3.7},
            {-12.0, 7.3},
            {-6.0, 2.4},
            {-6.0, 6.0},
            {0.0, 7.0},
            {0.0, 10.6},
            {5.0, 5.4},
            {5.0, 11.0},
            {-16.0, 11.5},
            {-10.0, -4.0},
            {-3.0, 13.0},
            {9.0, 13.5},
            {13.0, 2.0},
            {17.0, -2.5}
        };

        std::vector<std::pair<double, double>>obstacles_two_trees_trap = {
                {2.0, 4.5},   // Left V tree
                {2.0, 7.5},   // Right V tree
                
                // Main trap
                {4.0, 5.0},   // Left tree
                {4.0, 7.0},   // Right tree
                {6.0, 6.0},   // Center tree (creates local minimum)
                {8.0, 5.5}    // Front
        };      

           std::vector<std::pair<double, double>>obstacles_front_wall = {
            {6.0, 2.0},
            {6.0, 3.3},
            {6.0, 4.6},
            {6.0, 5.9},
            {6.0, 7.2},
            {6.0, 8.5},
            {6.0, 9.8}
        };

        obstacles_ = obstacles_medium_forest;

        openPathLoggingFiles(current_run_);

        // ------------------------------------------------------
        // Oscillation detector - physical, progress-aware safety net.
        // ------------------------------------------------------
        // Command sign changes are only counted when the UAV has also moved a
        // meaningful distance across the planned path. A failure additionally
        // requires poor goal progress and confinement to one local region, so
        // legitimate left/right turns through a cluttered environment are not
        // classified as a local-minimum oscillation.
        nh_.param("oscillation_vy_sign_threshold",
                  vy_sign_threshold_,
                  0.50);
        nh_.param("oscillation_min_lateral_reversal_distance",
                  minimum_lateral_reversal_distance_,
                  0.50);
        nh_.param("oscillation_limit",
                  oscillation_limit_,
                  5);
        nh_.param("oscillation_failure_time_s",
                  oscillation_failure_time_,
                  12.0);
        nh_.param("oscillation_memory_time_s",
                  oscillation_memory_time_,
                  4.0);
        nh_.param("oscillation_trap_radius_m",
                  oscillation_trap_radius_,
                  2.0);
        nh_.param("oscillation_min_goal_progress_m",
                  oscillation_minimum_goal_progress_,
                  1.0);

        prev_vy_sign_ = 0.0;
        last_accepted_reversal_lateral_ = 0.0;
        have_last_accepted_reversal_lateral_ = false;
        oscillation_count_ = 0;
        total_oscillation_count_ = 0;
        oscillation_failure_ = false;
        oscillation_active_ = false;
        oscillation_episode_start_time_ = 0.0;
        last_oscillation_time_ = 0.0;
        oscillation_episode_start_x_ = 0.0;
        oscillation_episode_start_y_ = 0.0;
        oscillation_episode_start_distance_to_goal_ = 0.0;

        // Progress detector - diagnostic/logging only in this version.
        progress_poor_ = false;
        progress_rate_ = 0.0;
        prev_distance_to_goal_ = 0.0;
        progress_timer_start_ = 0.0;
        progress_window_ = 2.0;
        min_progress_rate_ = 0.2;

        // -------------------------
        // Mission / flight settings
        // -------------------------
        // start_x_ = -19.0;
        // start_y_ = 0; //-3
        // goal_x_ = 8.0;
        // goal_y_ = 9.5;

        //zigzag
        start_x_ = -19.0;
        start_y_ = -5; //-3
        goal_x_ = 14.0;
        goal_y_ = 6;
        // goal_x_ = 8.0;
        // goal_y_ = 9.5;
        
        //U SHAPE
        // start_x_ = -2.0;//-2.0;
        // start_y_ = 6.0;//5.0;
        // goal_x_ = 14.0; //12.0;
        // goal_y_ = 6; //5.0;
        
        //TWO TREES
        // start_x_ = 0;
        // start_y_ = 6;
        // goal_x_ = 12.0;
        // goal_y_ = 6;

        goal_z_ = 1.50;

        // Forward velocity settings [m/s]
        forward_speed_ = 10; //sparse 10
        vx_command_ = 0.0;
        max_forward_acceleration_ = 4.0;
        max_forward_deceleration_ = 10.0;
        max_forward_speed_ = 25.0;
        max_lateral_speed_ = 10.0; //sparse 10
        max_vertical_speed_ = 0.5;

        // Fallback lateral nudge when danger is centred in the image (neither
        // side clearly worse) - keeps committing to the last meaningful side
        // instead of dithering near zero.
        default_avoid_y_speed_ = 1.2; //0.8
        danger_ttc_ = 2.0;
        last_avoid_direction_ = 1; //1

        // Final lateral command smoothing. Kept light - the heavy lifting
        // against noise now happens on repulsive_y_ itself (see below), so
        // this only needs to remove residual per-tick jitter.
        vy_filtered_ = 0.0;
        vy_alpha_ = 0.4;

        // APF gains
        k_att_y_ = 0.8;   // used only if use_path_controller_ is false
        k_rep_y_ = 1.3; //1.3
        k_z_ = 1.0;

        // ------------------------------------------------------
        // Longitudinal visual repulsion.
        // ------------------------------------------------------
        // The nominal forward target is the attractive x component. A frontal
        // VPF risk produces a bounded negative x repulsion that can cancel, but
        // never exceed, that attractive target:
        //
        //   vx_resultant = vx_attraction + vx_repulsion,  vx_resultant >= 0.
        //
        // This term is independent of rotational APF. Rotational APF is only
        // responsible for adding a tangential escape direction after the
        // longitudinal resultant has remained small for a short interval.
        nh_.param("enable_longitudinal_repulsion",
                  enable_longitudinal_repulsion_,
                  true);
        nh_.param("k_longitudinal_repulsion",
                  k_longitudinal_repulsion_,
                  0.9);
        nh_.param("longitudinal_repulsion_filter_alpha",
                  longitudinal_repulsion_filter_alpha_,
                  0.30);
        nh_.param("longitudinal_frontal_exponent",
                  longitudinal_frontal_exponent_,
                  2.0);
        nh_.param("max_forward_deceleration",
                  max_forward_deceleration_,
                  10.0);

        longitudinal_frontal_risk_filtered_ = 0.0;
        last_forward_attraction_target_ = 0.0;
        last_longitudinal_frontal_risk_ = 0.0;
        last_longitudinal_repulsive_vx_ = 0.0;
        last_forward_resultant_target_ = 0.0;

        // ------------------------------------------------------
        // Rotational APF: tangential escape in the horizontal X-Y plane.
        // ------------------------------------------------------
        // Activation is now tied to a persistent longitudinal local minimum:
        // the visual longitudinal repulsion has nearly cancelled the forward
        // attraction while a coherent frontal obstacle remains present.
        // Once active, the selected CW/CCW direction is locked until obstacle
        // influence decreases for the release delay.
        //
        // Set _enable_rotational_apf:=false for the baseline comparison. The
        // longitudinal repulsion remains active in that baseline.
        nh_.param("enable_rotational_apf",
                  enable_rotational_apf_,
                  true);
        nh_.param("rotational_trigger_on_trap",
                  rotational_trigger_on_trap_,
                  true);
        nh_.param("k_rotational",
                  k_rotational_,
                  0.8);
        nh_.param("rotational_activation_risk",
                  rotational_activation_risk_,
                  0.20);
        nh_.param("rotational_maintain_risk",
                  rotational_maintain_risk_,
                  0.10);
        nh_.param("rotational_start_forward_speed",
                  rotational_start_forward_speed_,
                  0.50);
        nh_.param("rotational_start_delay_s",
                  rotational_start_delay_s_,
                  0.30);
        nh_.param("rotational_direction_deadband_rad",
                  rotational_direction_deadband_rad_,
                  0.15);
        nh_.param("rotational_min_speed",
                  rotational_min_speed_,
                  0.2);
        nh_.param("rotational_max_speed",
                  rotational_max_speed_,
                  1.0);
        // Blend a modest obstacle-away radial component into the tangential
        // rotational escape. A pure tangent can preserve an already-small
        // obstacle radius and scrape the pole; this term actively increases
        // clearance while retaining the around-obstacle behaviour.
        nh_.param("rotational_radial_escape_weight",
                  rotational_radial_escape_weight_,
                  0.60);
        // While rotational APF owns the escape, prevent the nominal 10 m/s
        // forward attraction from overwhelming the <=1 m/s escape vector.
        nh_.param("rotational_forward_speed_cap_mps",
                  rotational_forward_speed_cap_,
                  1.00);
        // Hand control back to explicit gap guidance once a genuinely safe gap
        // has remained available for a short continuous period.
        nh_.param("rotational_safe_gap_release_delay_s",
                  rotational_safe_gap_release_delay_s_,
                  0.30);
        nh_.param("rotational_filter_alpha",
                  rotational_filter_alpha_,
                  0.25);
        nh_.param("rotational_release_delay_s",
                  rotational_release_delay_s_,
                  0.75);

        rotational_apf_active_ = false;
        rotational_apf_used_ = false;
        rotational_direction_ = 0; // +1 clockwise, -1 counter-clockwise
        rotational_activation_count_ = 0;
        rotational_last_request_time_ = 0.0;
        rotational_blockage_active_ = false;
        rotational_blockage_start_time_ = 0.0;
        last_rotational_blockage_duration_ = 0.0;
        rotational_safe_gap_reacquire_active_ = false;
        rotational_safe_gap_reacquire_start_time_ = 0.0;

        last_visual_obstacle_x_body_ = 1.0;
        last_visual_obstacle_y_body_ = 0.0;
        last_rotational_risk_strength_ = 0.0;
        last_rotational_vx_body_ = 0.0;
        last_rotational_vy_body_ = 0.0;
        rotational_vx_filtered_ = 0.0;
        rotational_vy_filtered_ = 0.0;

        last_image_width_ = 640;
        last_image_height_ = 480;

        // ------------------------------------------------------
        // Repulsion signal smoothing - applied to repulsive_y_ itself, right
        // after it's computed from the VPF grid, BEFORE any gain or mixing.
        // This is the correct place to fight per-frame vision noise: fixing
        // it here means every downstream consumer (the gain, the final
        // command filter, the direction-hysteresis) all see a clean signal.
        // ------------------------------------------------------
        repulsive_y_filtered_ = 0.0;
        repulsive_alpha_ = 0.3;

        // Visual Potential Field grid
        vpf_grid_cols_ = 16;
        vpf_grid_rows_ = 8;
        vpf_risk_ = cv::Mat::zeros(vpf_grid_rows_, vpf_grid_cols_, CV_32F);
        vpf_ttc_weight_ = 1.0;
        vpf_flow_weight_ = 0.0;  // raw flow does not create danger


        // ------------------------------------------------------
        // Column-wise VPF free-gap guidance for dense clutter.
        // ------------------------------------------------------
        // Instead of reducing the entire VPF to one left-minus-right value,
        // preserve horizontal structure by computing one danger value per image
        // column. Consecutive low-risk columns form candidate free gaps. The
        // selected gap is biased mildly toward the goal and toward the previously
        // selected gap, so the vehicle commits to a corridor instead of dithering.
        // If no sufficiently wide safe gap exists, the rotational APF is requested.
        nh_.param("enable_gap_guidance",
                  enable_gap_guidance_,
                  true);
        nh_.param("gap_safe_column_risk_threshold",
                  gap_safe_column_risk_threshold_,
                  0.12);
        nh_.param("gap_activation_risk_threshold",
                  gap_activation_risk_threshold_,
                  0.15);
        // Do not enter rotational escape just because sparse LK briefly fails
        // to expose a >=4-column gap.  A no-gap event must also represent a
        // genuinely strong obstacle field.  This reduces unnecessary 0.8-1.0 m/s
        // rotational episodes in mild clutter without changing the heading controller.
        nh_.param("gap_rotational_trigger_risk_threshold",
                  gap_rotational_trigger_risk_threshold_,
                  0.45);
        nh_.param("gap_min_width_cols",
                  gap_min_width_cols_,
                  4);
        nh_.param("gap_hold_time_s",
                  gap_hold_time_s_,
                  1.00);
        nh_.param("gap_release_delay_s",
                  gap_release_delay_s_,
                  0.80);
        // Keep a recent "no safe gap" request alive long enough for the
        // rotational APF confirmation timer to complete. Without this latch,
        // one-frame reappearance of a sparse visual gap repeatedly resets the
        // rotational start timer and the UAV can reach the obstacle before
        // tangential escape ever activates.
        nh_.param("gap_no_safe_latch_s",
                  gap_no_safe_latch_s_,
                  0.60);
        nh_.param("gap_goal_bias_weight",
                  gap_goal_bias_weight_,
                  0.12);
        // Direct goal attraction used during gap following. The goal heading is
        // first clamped INSIDE the selected visual gap and only then blended
        // with the gap centre, so goal attraction cannot pull the UAV through a
        // visually occupied column.
        nh_.param("gap_goal_heading_blend",
                  gap_goal_heading_blend_,
                  0.25);
        // A small conventional attractive component is retained when danger is
        // active but no safe gap is available / rotational escape is active.
        nh_.param("avoidance_attraction_scale",
                  avoidance_attraction_scale_,
                  0.15);
        nh_.param("rotational_attraction_scale",
                  rotational_attraction_scale_,
                  0.10);
        nh_.param("gap_continuity_weight",
                  gap_continuity_weight_,
                  0.45);
        nh_.param("gap_width_bonus_weight",
                  gap_width_bonus_weight_,
                  0.18);
        nh_.param("gap_max_forward_speed_mps",
                  gap_max_forward_speed_,
                  5.0);
        nh_.param("gap_min_forward_speed_mps",
                  gap_min_forward_speed_,
                  1.5);
        // Stay farther from the visual edges of a selected corridor. Sparse LK
        // often marks pole edges rather than their whole silhouette, so the
        // old 0.35-column margin could still skim a pole boundary.
        nh_.param("gap_edge_margin_cols",
                  gap_edge_margin_cols_,
                  0.75);

        resetGapGuidanceState();

        // Risk shaping (baseline inverse-TTC by default; kept switchable).
        //   0 = 1/(TTC+eps)         1 = sqrt(1/(TTC+eps))
        //   2 = (1/(TTC+eps))^a     3 = logistic threshold
        risk_shaping_mode_ = 0;
        inverse_ttc_epsilon_ = 0.05;
        risk_power_alpha_ = 1.5;
        logistic_beta_ = 1.5;
        logistic_threshold_ = 5.0;
        max_ttc_risk_ = 5.0;

        enable_visualization_ = true;

        use_gaussian_blur_ = false;
        gaussian_kernel_size_ = 5;
        gaussian_sigma_ = 1.2;
        if (gaussian_kernel_size_ <= 0 || gaussian_kernel_size_ % 2 == 0)
        {
            ROS_WARN("Invalid Gaussian kernel size %d. Falling back to 5.", gaussian_kernel_size_);
            gaussian_kernel_size_ = 5;
        }

        first_image_time_ = -1.0;
        last_image_time_ = -1.0;
        previous_image_time_for_dt_ = -1.0;
        image_count_ = 0;
        min_image_dt_ = std::numeric_limits<double>::max();
        max_image_dt_ = 0.0;
        processing_time_sum_ms_ = 0.0;
        max_processing_time_ms_ = 0.0;
        processing_count_ = 0;

        repulsive_y_ = 0.0;
        left_danger_ = 0.0;
        right_danger_ = 0.0;
        tracked_features_ = 0;
        ttc_filtered_ = 0.0;
        ttc_max_useful_ = 15.0;
        ttc_miss_count_ = 0;
        ttc_miss_limit_ = 5;

        // ------------------------------------------------------
        // Dedicated frontal TTC channel.
        // ------------------------------------------------------
        // Global TTC is a median over the whole trusted image and can be
        // dominated by farther obstacles in dense clutter.  This second TTC
        // estimate uses only the central image corridor and a lower quantile,
        // so a pole directly on the flight axis remains safety-critical.
        nh_.param("frontal_ttc_danger_s",
                  frontal_ttc_danger_,
                  2.5);
        nh_.param("frontal_ttc_min_x_fraction",
                  frontal_ttc_min_x_fraction_,
                  0.375);
        nh_.param("frontal_ttc_max_x_fraction",
                  frontal_ttc_max_x_fraction_,
                  0.625);
        nh_.param("frontal_ttc_quantile",
                  frontal_ttc_quantile_,
                  0.25);
        nh_.param("frontal_ttc_filter_alpha",
                  frontal_ttc_filter_alpha_,
                  0.30);
        nh_.param("frontal_ttc_min_features",
                  frontal_ttc_min_features_,
                  3);
        nh_.param("frontal_ttc_miss_limit",
                  frontal_ttc_miss_limit_,
                  5);

        frontal_ttc_filtered_ = 0.0;
        have_frontal_ttc_filtered_ = false;
        frontal_ttc_miss_count_ = 0;

        // ------------------------------------------------------
        // Trusted image band for the first controlled VPF test.
        // ------------------------------------------------------
        floor_row_min_frac_ = 0.35;
        floor_row_max_frac_ = 0.80;

        // LK track validation.
        max_lk_error_ = 20.0;
        max_flow_pixels_ = 40.0;

        // Dynamic-FOE VPF validation parameters.
        minimum_vpf_flow_pixels_ = 0.10;
        minimum_radial_ratio_ = 0.50;
        local_danger_ttc_ = 5.0;
        minimum_cell_support_ = 2;

        // Feature detection / LK parameters
        max_corners_ = 150;
        quality_level_ = 0.005;
        min_distance_ = 5.0;
        block_size_ = 7;
        lk_win_size_ = cv::Size(21, 21);
        lk_max_level_ = 3;
        lk_criteria_ = cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 30, 0.01);

        vel_pub_ = nh_public_.advertise<geometry_msgs::TwistStamped>("/uav1/command/twist", 10);
        if (enable_visualization_)
        {
            vis_pub_ = nh_public_.advertise<sensor_msgs::Image>("/uav1/optical_flow/image", 1);
        }
        image_sub_ = nh_public_.subscribe("/uav1/front_camera/image_raw", 1,
                                          &OnlineLKAPFVelocity::imageCallback, this);
        pose_sub_ = nh_public_.subscribe("/uav1/ground_truth/state", 1,
                                         &OnlineLKAPFVelocity::poseCallback, this);

        gazebo_reset_client_ = nh_public_.serviceClient<std_srvs::Empty>(gazebo_reset_service_);

        enableMotors();

        control_timer_ = nh_public_.createTimer(ros::Duration(0.01),
                                                &OnlineLKAPFVelocity::controlLoop, this);

        ROS_INFO("Online LK + VPF + APF (lateral-only) node started.");
        ROS_INFO("Repeated experiment: target_runs=%d, no-movement timeout=%.1f s, movement threshold=%.2f m",
                 target_runs_, no_movement_timeout_s_, movement_threshold_m_);
        ROS_INFO("Gazebo reset service: %s", gazebo_reset_service_.c_str());
        ROS_INFO("Post-reset stabilization time: %.1f s", stabilization_time_s_);
        ROS_INFO("Look-ahead path recovery: distance=%.2f m, max lateral recovery=%.2f m/s, k_path_y=%.2f",
                 path_lookahead_distance_,
                 max_path_recovery_speed_,
                 k_path_y_);
        ROS_INFO("Velocity-heading control: %s, k=%.2f, max rate=%.2f rad/s, max body vy=%.2f m/s",
                 enable_velocity_heading_control_ ? "enabled" : "disabled",
                 k_velocity_heading_,
                 max_velocity_heading_rate_,
                 max_body_lateral_speed_);
        ROS_INFO("Dynamic FOE perception: body vx>%.2f m/s, FOE from measured body translation; legacy heading/lateral centre-FOE gates are bypassed, TTC hold=%.2f s, risk decay=%.3f",
                 dynamic_foe_min_forward_body_speed_,
                 centre_foe_ttc_hold_s_,
                 centre_foe_risk_decay_);
        ROS_INFO("Column-gap guidance: %s, safe_col_risk<=%.2f, activation_risk>=%.2f, min_width=%d cols, nominal_gap_speed=[%.2f, %.2f] m/s (safety cap may reduce it)",
                  enable_gap_guidance_ ? "enabled" : "disabled",
                  gap_safe_column_risk_threshold_,
                  gap_activation_risk_threshold_,
                  gap_min_width_cols_,
                  gap_min_forward_speed_,
                  gap_max_forward_speed_);
        ROS_INFO("Yaw-flow compensation: %s (fx=%.1f px, fy=%.1f px, sign=%.1f)",
                 enable_yaw_flow_compensation_ ? "enabled" : "disabled",
                 camera_fx_pixels_,
                 camera_fy_pixels_,
                 yaw_flow_sign_);
        ROS_INFO("Rotational APF: %s, trigger=%s, k_rot=%.2f, speed=[%.2f, %.2f] m/s",
                 enable_rotational_apf_ ? "enabled" : "disabled",
                 rotational_trigger_on_trap_ ? "trap-only" : "repulsion-active",
                 k_rotational_,
                 rotational_min_speed_,
                 rotational_max_speed_);
        ROS_INFO("Longitudinal repulsion: %s, k=%.2f, filter alpha=%.2f, frontal exponent=%.2f",
                  enable_longitudinal_repulsion_ ? "enabled" : "disabled",
                  k_longitudinal_repulsion_,
                  longitudinal_repulsion_filter_alpha_,
                  longitudinal_frontal_exponent_);
        ROS_INFO("Frontal TTC safety channel: moving FOE-centred corridor width from fractions [%.3f, %.3f], q=%.2f, danger<%.2f s, min_features=%d",
                 frontal_ttc_min_x_fraction_,
                 frontal_ttc_max_x_fraction_,
                 frontal_ttc_quantile_,
                 frontal_ttc_danger_,
                 frontal_ttc_min_features_);
        ROS_INFO("Rotational local-minimum trigger: resultant vx<=%.2f m/s for %.2f s, maintain risk>=%.2f",
                  rotational_start_forward_speed_,
                  rotational_start_delay_s_,
                  rotational_maintain_risk_);
        ROS_INFO("Forward acceleration/deceleration limits: +%.2f / -%.2f m/s^2",
                  max_forward_acceleration_,
                  max_forward_deceleration_);
        ROS_INFO("Oscillation detector: |vy|>=%.2f m/s, lateral excursion>=%.2f m, "
                 "limit=%d, failure time=%.1f s, memory=%.1f s, trap radius=%.2f m, "
                 "minimum goal progress=%.2f m",
                 vy_sign_threshold_,
                 minimum_lateral_reversal_distance_,
                 oscillation_limit_,
                 oscillation_failure_time_,
                 oscillation_memory_time_,
                 oscillation_trap_radius_,
                 oscillation_minimum_goal_progress_);
    }

    ~OnlineLKAPFVelocity()
    {
        closePathLoggingFiles();
    }

    void enableMotors()
    {
        ROS_INFO("Waiting for motor service...");
        try
        {
            if (ros::service::waitForService("/uav1/enable_motors", ros::Duration(10.0)))
            {
                ros::ServiceClient client = nh_public_.serviceClient<hector_uav_msgs::EnableMotors>("/uav1/enable_motors");
                hector_uav_msgs::EnableMotors srv;
                srv.request.enable = true;
                if (client.call(srv))
                {
                    ROS_INFO("Motors enabled.");
                }
                else
                {
                    ROS_WARN("Could not call /uav1/enable_motors service.");
                }
            }
            else
            {
                ROS_WARN("Motor service not available within timeout.");
            }
        }
        catch (const std::exception& e)
        {
            ROS_WARN("Could not enable motors automatically: %s", e.what());
        }
    }

    void openPathLoggingFiles(int run_number)
    {
        closePathLoggingFiles();

        std::ostringstream suffix;
        suffix << std::setw(3) << std::setfill('0') << run_number;

        const std::string path_filename = "/root/catkin_ws/uav_path_attempt_" + suffix.str() + ".csv";
        const std::string obstacle_filename = "/root/catkin_ws/obstacles_attempt_" + suffix.str() + ".csv";

        path_file_.open(path_filename);
        obstacle_file_.open(obstacle_filename);

        if (!path_file_.is_open())
        {
            ROS_WARN("Could not open %s for writing.", path_filename.c_str());
            path_files_opened_ = false;
            return;
        }
        if (!obstacle_file_.is_open())
        {
            ROS_WARN("Could not open %s for writing.", obstacle_filename.c_str());
            path_file_.close();
            path_files_opened_ = false;
            return;
        }

        path_file_
            << "time,x,y,z,yaw,"
            << "vx,vy,vz,ttc,forward_attraction_target_mps,"
            << "longitudinal_frontal_risk,longitudinal_repulsive_vx_mps,"
            << "forward_resultant_target_mps,current_obstacle_clearance_m,"
            << "repulsive_y_raw,repulsive_y_filtered,left_danger,right_danger,"
            << "rotational_active,rotational_direction,"
            << "rotational_vx_body,rotational_vy_body,rotational_risk_strength,"
            << "rotational_blockage_duration_s,"
            << "y_path_ref,cross_track_error,v_path_y,v_goal_y,"
            << "tracked_features,progress_poor,collision_count,"
            << "body_vx,body_vy,body_vz,desired_heading_rad,heading_error_rad,"
            << "centre_foe_valid,motion_alignment_scale,"
            << "gap_guidance_active,gap_available,gap_escape_required,"
            << "gap_start_col,gap_end_col,gap_center_col,gap_width_cols,"
            << "gap_heading_offset_rad,gap_mean_risk,max_column_risk,"
            << "frontal_ttc_s,frontal_ttc_valid,frontal_danger,"
            << "dynamic_foe_x_px,dynamic_foe_y_px,dynamic_foe_valid,"
            << "frontal_corridor_x_min_px,frontal_corridor_x_max_px\n";

        obstacle_file_ << "x,y,radius\n";
        for (const auto& obs : obstacles_)
        {
            obstacle_file_ << obs.first << "," << obs.second << "," << obstacle_radius_ << "\n";
        }
        obstacle_file_.close();
        path_files_opened_ = true;

        ROS_INFO("Run %d CSV logging enabled:", run_number);
        ROS_INFO("  %s", path_filename.c_str());
        ROS_INFO("  %s", obstacle_filename.c_str());
    }

    void closePathLoggingFiles()
    {
        if (path_file_.is_open())
        {
            path_file_.flush();
            path_file_.close();
        }
        if (obstacle_file_.is_open())
        {
            obstacle_file_.close();
        }
    }

    double getAverageImageRate() const
    {
        if (image_count_ > 1 && last_image_time_ > first_image_time_)
        {
            return static_cast<double>(image_count_ - 1) / (last_image_time_ - first_image_time_);
        }
        return 0.0;
    }

    double getAverageProcessingTimeMs() const
    {
        if (processing_count_ > 0)
        {
            return processing_time_sum_ms_ / static_cast<double>(processing_count_);
        }
        return 0.0;
    }

    double getMinimumImageInterval() const
    {
        return (image_count_ > 1 && min_image_dt_ < std::numeric_limits<double>::max()) ? min_image_dt_ : 0.0;
    }

    double getMaximumImageInterval() const
    {
        return (image_count_ > 1) ? max_image_dt_ : 0.0;
    }

    std::string riskShapingName() const
    {
        switch (risk_shaping_mode_)
        {
            case 0: return "inverse_ttc";
            case 1: return "sqrt_inverse_ttc";
            case 2: return "power_inverse_ttc";
            case 3: return "logistic_ttc";
            default: return "unknown";
        }
    }

    double shapeTTCRisk(double ttc) const
    {
        const double safe_ttc = std::max(ttc, inverse_ttc_epsilon_);
        const double inv_ttc = 1.0 / safe_ttc;
        double shaped_risk = inv_ttc;

        switch (risk_shaping_mode_)
        {
            case 0: shaped_risk = inv_ttc; break;
            case 1: shaped_risk = std::sqrt(inv_ttc); break;
            case 2: shaped_risk = std::pow(inv_ttc, risk_power_alpha_); break;
            case 3:
                shaped_risk = 1.0 / (1.0 + std::exp(logistic_beta_ * (safe_ttc - logistic_threshold_)));
                break;
            default: shaped_risk = inv_ttc; break;
        }

        return clamp(shaped_risk, 0.0, max_ttc_risk_);
    }

    bool isTrustedObstaclePoint(
        const cv::Mat& color_frame,
        double x,
        double y,
        int h) const
    {
        // Geometry-only row-band filter.
        if (y < h * floor_row_min_frac_ ||
            y > h * floor_row_max_frac_)
        {
            return false;
        }

        const int px = static_cast<int>(x);
        const int py = static_cast<int>(y);

        if (px < 0 || px >= color_frame.cols ||
            py < 0 || py >= color_frame.rows)
        {
            return false;
        }

        return true;
    }

    void appendExperimentResults(const std::string& termination_reason,
                                 double final_x, double final_y, double final_z,
                                 bool success)
    {
        const bool file_exists = static_cast<bool>(std::ifstream(results_file_path_));
        bool write_header = true;
        if (file_exists)
        {
            std::ifstream in(results_file_path_, std::ios::ate);
            write_header = (!in.is_open() || in.tellg() == 0);
        }

        std::ofstream results_file(results_file_path_, std::ios::app);
        if (!results_file.is_open())
        {
            ROS_WARN("Could not open %s for appending experiment results.", results_file_path_.c_str());
            return;
        }

        if (write_header)
        {
            results_file
                << "run_number,completed_runs,successful_runs,success,ros_time,termination,"
                << "final_x,final_y,final_z,path_length_m,execution_time_s,collisions,"
                << "minimum_clearance_including_collisions_m,minimum_collision_free_clearance_m,"
                << "overall_closest_obstacle_x,overall_closest_obstacle_y,"
                << "safe_closest_obstacle_x,safe_closest_obstacle_y,"
                << "tracked_features,oscillation_count,total_oscillations,"
                << "progress_poor,progress_rate_mps,path_controller_used,"
                << "final_y_path_ref,final_cross_track_error,k_path_y,k_goal_y,"
                << "risk_shaping_function,risk_shaping_mode,risk_power_alpha,"
                << "logistic_beta,logistic_threshold,final_yaw_rad,"
                << "image_count,avg_image_rate_hz,min_image_interval_s,max_image_interval_s,"
                << "avg_processing_time_ms,max_processing_time_ms,"
                << "forward_speed_mps,k_att_y,k_rep_y,repulsive_alpha,danger_ttc_s,"
                << "max_lateral_speed_mps,max_forward_deceleration_mps2,"
                << "longitudinal_repulsion_enabled,k_longitudinal_repulsion,"
                << "longitudinal_repulsion_filter_alpha,longitudinal_frontal_exponent,"
                << "rotational_apf_enabled,rotational_apf_used,rotational_activation_count,"
                << "rotational_trigger_on_trap,k_rotational,rotational_start_forward_speed,"
                << "rotational_start_delay_s,rotational_maintain_risk,"
                << "start_x,start_y,goal_x,goal_y,goal_z\n";
        }

        results_file << std::fixed << std::setprecision(4);

        const double overall_min_clearance_csv =
            (minimum_obstacle_clearance_ < std::numeric_limits<double>::max())
                ? minimum_obstacle_clearance_ : std::numeric_limits<double>::quiet_NaN();
        const double collision_free_min_clearance_csv =
            (minimum_collision_free_clearance_ < std::numeric_limits<double>::max())
                ? minimum_collision_free_clearance_ : std::numeric_limits<double>::quiet_NaN();

        results_file
            << current_run_ << "," << completed_runs_ << "," << successful_runs_ << ","
            << boolText(success) << "," << ros::Time::now().toSec() << "," << termination_reason << ","
            << final_x << "," << final_y << "," << final_z << ","
            << path_length_ << "," << getMissionExecutionTime() << "," << collision_count_ << ","
            << overall_min_clearance_csv << "," << collision_free_min_clearance_csv << ","
            << minimum_clearance_obstacle_x_ << "," << minimum_clearance_obstacle_y_ << ","
            << minimum_safe_clearance_obstacle_x_ << "," << minimum_safe_clearance_obstacle_y_ << ","
            << tracked_features_ << "," << oscillation_count_ << "," << total_oscillation_count_ << ","
            << boolText(progress_poor_) << "," << progress_rate_ << "," << boolText(use_path_controller_) << ","
            << last_y_ref_ << "," << last_cross_track_error_ << "," << k_path_y_ << "," << k_goal_y_ << ","
            << riskShapingName() << "," << risk_shaping_mode_ << "," << risk_power_alpha_ << ","
            << logistic_beta_ << "," << logistic_threshold_ << "," << current_yaw_ << ","
            << image_count_ << "," << getAverageImageRate() << "," << getMinimumImageInterval() << ","
            << getMaximumImageInterval() << "," << getAverageProcessingTimeMs() << "," << max_processing_time_ms_ << ","
            << forward_speed_ << "," << k_att_y_ << "," << k_rep_y_ << "," << repulsive_alpha_ << "," << danger_ttc_ << ","
            << max_lateral_speed_ << ","
            << max_forward_deceleration_ << ","
            << boolText(enable_longitudinal_repulsion_) << ","
            << k_longitudinal_repulsion_ << ","
            << longitudinal_repulsion_filter_alpha_ << ","
            << longitudinal_frontal_exponent_ << ","
            << boolText(enable_rotational_apf_) << ","
            << boolText(rotational_apf_used_) << ","
            << rotational_activation_count_ << ","
            << boolText(rotational_trigger_on_trap_) << ","
            << k_rotational_ << ","
            << rotational_start_forward_speed_ << ","
            << rotational_start_delay_s_ << ","
            << rotational_maintain_risk_ << ","
            << start_x_ << "," << start_y_ << "," << goal_x_ << "," << goal_y_ << "," << goal_z_
            << "\n";

        results_file.close();
        ROS_INFO("Experiment summary appended to %s", results_file_path_.c_str());
    }

    void logPathSample()
    {
        if (!path_files_opened_ || !path_file_.is_open() || !takeoff_done_)
        {
            return;
        }

        path_file_
            << ros::Time::now().toSec() << ","
            << current_pose_.position.x << "," << current_pose_.position.y << "," << current_pose_.position.z << ","
            << current_yaw_ << ","
            << last_vx_ << "," << last_vy_ << "," << last_vz_ << ",";

        if (have_ttc_filtered_) { path_file_ << ttc_filtered_; } else { path_file_ << "nan"; }
        path_file_
            << "," << last_forward_attraction_target_
            << "," << last_longitudinal_frontal_risk_
            << "," << last_longitudinal_repulsive_vx_
            << "," << last_forward_resultant_target_
            << ",";

        if (current_obstacle_clearance_ < std::numeric_limits<double>::max())
        {
            path_file_ << current_obstacle_clearance_;
        }
        else
        {
            path_file_ << "nan";
        }

        path_file_
            << "," << repulsive_y_ << "," << repulsive_y_filtered_ << ","
            << left_danger_ << "," << right_danger_ << ","
            << static_cast<int>(rotational_apf_active_) << ","
            << rotational_direction_ << ","
            << last_rotational_vx_body_ << ","
            << last_rotational_vy_body_ << ","
            << last_rotational_risk_strength_ << ","
            << last_rotational_blockage_duration_ << ","
            << last_y_ref_ << "," << last_cross_track_error_ << ","
            << last_v_path_y_ << "," << last_v_goal_y_ << ","
            << tracked_features_ << "," << static_cast<int>(progress_poor_) << ","
            << collision_count_ << ","
            << current_body_vx_ << "," << current_body_vy_ << "," << current_body_vz_ << ","
            << desired_heading_ << "," << last_heading_error_ << ","
            << static_cast<int>(centre_foe_model_valid_) << ","
            << last_motion_alignment_scale_ << ","
            << static_cast<int>(gap_guidance_active_) << ","
            << static_cast<int>(safe_gap_available_) << ","
            << static_cast<int>(gap_escape_required_) << ","
            << selected_gap_start_col_ << ","
            << selected_gap_end_col_ << ","
            << selected_gap_center_col_ << ","
            << selected_gap_width_cols_ << ","
            << selected_gap_heading_offset_rad_ << ","
            << selected_gap_mean_risk_ << ","
            << max_column_risk_ << ",";

        if (have_frontal_ttc_filtered_)
        {
            path_file_ << frontal_ttc_filtered_;
        }
        else
        {
            path_file_ << "nan";
        }

        const bool frontal_danger_now =
            have_frontal_ttc_filtered_ &&
            frontal_ttc_filtered_ < frontal_ttc_danger_;

        path_file_
            << "," << static_cast<int>(have_frontal_ttc_filtered_)
            << "," << static_cast<int>(frontal_danger_now)
            << "," << dynamic_foe_x_px_
            << "," << dynamic_foe_y_px_
            << "," << static_cast<int>(dynamic_foe_valid_)
            << "," << frontal_corridor_x_min_px_
            << "," << frontal_corridor_x_max_px_
            << "\n";
    }

    void poseCallback(const nav_msgs::OdometryConstPtr& msg)
    {
        current_pose_ = msg->pose.pose;
        current_yaw_ = yawFromQuaternion(current_pose_.orientation);

        // The gazebo_ros_p3d ground-truth plugin in this model publishes
        // WORLD-frame linear and angular velocity because <localTwist> is not
        // configured (its default is false). Convert both vectors from world
        // coordinates into the UAV/body frame before using them for:
        //   1) centre-FOE validity checking, and
        //   2) yaw-induced optical-flow compensation.
        //
        // The pose quaternion describes the body orientation in the world.
        // Therefore R_body_to_world is formed from the quaternion and its
        // transpose is used for the world-to-body vector conversion.
        const geometry_msgs::Quaternion& q_msg =
            current_pose_.orientation;

        const double q_norm =
            std::sqrt(
                q_msg.x * q_msg.x +
                q_msg.y * q_msg.y +
                q_msg.z * q_msg.z +
                q_msg.w * q_msg.w);

        double qx = 0.0;
        double qy = 0.0;
        double qz = 0.0;
        double qw = 1.0;

        if (q_norm > 1e-9)
        {
            qx = q_msg.x / q_norm;
            qy = q_msg.y / q_norm;
            qz = q_msg.z / q_norm;
            qw = q_msg.w / q_norm;
        }

        // Body-to-world rotation matrix.
        const double r00 = 1.0 - 2.0 * (qy * qy + qz * qz);
        const double r01 = 2.0 * (qx * qy - qw * qz);
        const double r02 = 2.0 * (qx * qz + qw * qy);

        const double r10 = 2.0 * (qx * qy + qw * qz);
        const double r11 = 1.0 - 2.0 * (qx * qx + qz * qz);
        const double r12 = 2.0 * (qy * qz - qw * qx);

        const double r20 = 2.0 * (qx * qz - qw * qy);
        const double r21 = 2.0 * (qy * qz + qw * qx);
        const double r22 = 1.0 - 2.0 * (qx * qx + qy * qy);

        const double measured_world_vx =
            msg->twist.twist.linear.x;
        const double measured_world_vy =
            msg->twist.twist.linear.y;
        const double measured_world_vz =
            msg->twist.twist.linear.z;

        // World -> body: v_body = R_body_to_world^T * v_world.
        current_body_vx_ =
            r00 * measured_world_vx +
            r10 * measured_world_vy +
            r20 * measured_world_vz;

        current_body_vy_ =
            r01 * measured_world_vx +
            r11 * measured_world_vy +
            r21 * measured_world_vz;

        current_body_vz_ =
            r02 * measured_world_vx +
            r12 * measured_world_vy +
            r22 * measured_world_vz;

        const double measured_world_wx =
            msg->twist.twist.angular.x;
        const double measured_world_wy =
            msg->twist.twist.angular.y;
        const double measured_world_wz =
            msg->twist.twist.angular.z;

        const double measured_body_wz =
            r02 * measured_world_wx +
            r12 * measured_world_wy +
            r22 * measured_world_wz;

        current_yaw_rate_ =
            yaw_rate_filter_alpha_ * measured_body_wz +
            (1.0 - yaw_rate_filter_alpha_) * current_yaw_rate_;

        ROS_INFO_THROTTLE(
            0.5,
            "TWIST_FRAME: yaw=%.1f deg world_v=(%.3f, %.3f, %.3f) "
            "body_v=(%.3f, %.3f, %.3f) body_wz=%.3f",
            current_yaw_ * 180.0 / M_PI,
            measured_world_vx,
            measured_world_vy,
            measured_world_vz,
            current_body_vx_,
            current_body_vy_,
            current_body_vz_,
            measured_body_wz);

        have_pose_ = true;

        if (mission_timer_started_ && !results_printed_ && !reached_goal_ && !oscillation_failure_)
        {
            const double x = current_pose_.position.x;
            const double y = current_pose_.position.y;
            const double z = current_pose_.position.z;

            if (!have_last_path_pose_)
            {
                last_path_x_ = x; last_path_y_ = y; last_path_z_ = z;
                have_last_path_pose_ = true;
            }
            else
            {
                const double dx = x - last_path_x_;
                const double dy = y - last_path_y_;
                const double dz = z - last_path_z_;
                path_length_ += std::sqrt(dx * dx + dy * dy + dz * dz);
                last_path_x_ = x; last_path_y_ = y; last_path_z_ = z;
            }
        }

        logPathSample();
    }


    bool computeDynamicFOE(
        const cv::Size& image_size,
        double& foe_x,
        double& foe_y) const
    {
        const double w = static_cast<double>(std::max(1, image_size.width));
        const double h = static_cast<double>(std::max(1, image_size.height));
        const double cx = 0.5 * w;
        const double cy = 0.5 * h;

        foe_x = cx;
        foe_y = cy;

        // A finite translational FOE requires a positive camera-forward
        // component.  Unlike the old centre-FOE gate, no restriction is imposed
        // on heading error or |vy|/vx: those terms move the FOE rather than
        // invalidating the perception update.
        if (!have_pose_ ||
            current_body_vx_ <= dynamic_foe_min_forward_body_speed_)
        {
            return false;
        }

        const double fx =
            (camera_fx_pixels_ > 1.0) ? camera_fx_pixels_ : cx;
        const double fy =
            (camera_fy_pixels_ > 1.0) ? camera_fy_pixels_ : cx;

        foe_x =
            cx - fx * current_body_vy_ / current_body_vx_;
        foe_y =
            cy - fy * current_body_vz_ / current_body_vx_;

        if (!std::isfinite(foe_x) || !std::isfinite(foe_y))
        {
            foe_x = cx;
            foe_y = cy;
            return false;
        }

        // Extremely distant FOEs occur as vx approaches zero.  They are valid in
        // ideal projective geometry but become numerically fragile for sparse LK
        // flow.  Reject only truly extreme cases; ordinary off-image FOEs during
        // crabbing are intentionally retained.
        const double max_offset =
            std::max(1.0, dynamic_foe_max_offset_image_widths_) * w;

        if (std::abs(foe_x - cx) > max_offset ||
            std::abs(foe_y - cy) > max_offset)
        {
            return false;
        }

        return true;
    }

    void decayCentreFOEVisualState(double now)
    {
        // A dynamic FOE is unavailable only when forward translation is too
        // small (or numerically degenerate). Preserve the previous obstacle
        // episode briefly, then release it smoothly.
        if (!vpf_risk_.empty())
        {
            vpf_risk_ *=
                static_cast<float>(
                    clamp(centre_foe_risk_decay_, 0.0, 1.0));
        }

        computeVPFRepulsionY(
            repulsive_y_,
            left_danger_,
            right_danger_);

        updateColumnGapSelection(now);

        repulsive_y_filtered_ =
            repulsive_alpha_ * repulsive_y_ +
            (1.0 - repulsive_alpha_) * repulsive_y_filtered_;

        const bool within_ttc_hold =
            have_last_centre_foe_valid_time_ &&
            (now - last_centre_foe_valid_time_) <=
                centre_foe_ttc_hold_s_;

        if (!within_ttc_hold)
        {
            have_ttc_filtered_ = false;
            ttc_filtered_ = 0.0;
            ttc_miss_count_ = 0;

            have_frontal_ttc_filtered_ = false;
            frontal_ttc_filtered_ = 0.0;
            frontal_ttc_miss_count_ = 0;
        }
    }

    void imageCallback(const sensor_msgs::ImageConstPtr& msg)
    {
        if (resetting_world_ || stabilizing_after_reset_ || attempt_finished_ || experiment_complete_)
        {
            return;
        }

        const auto callback_start = std::chrono::high_resolution_clock::now();
        auto finish_processing_measurement = [this, &callback_start]()
        {
            const auto callback_end = std::chrono::high_resolution_clock::now();
            const double processing_ms =
                std::chrono::duration<double, std::milli>(callback_end - callback_start).count();
            processing_time_sum_ms_ += processing_ms;
            processing_count_++;
            if (processing_ms > max_processing_time_ms_) { max_processing_time_ms_ = processing_ms; }
        };

        cv_bridge::CvImagePtr cv_ptr;
        try
        {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        }
        catch (const cv_bridge::Exception& e)
        {
            ROS_WARN("cv_bridge exception: %s", e.what());
            finish_processing_measurement();
            return;
        }

        cv::Mat frame = cv_ptr->image;
        last_image_width_ = frame.cols;
        last_image_height_ = frame.rows;

        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        if (use_gaussian_blur_)
        {
            cv::GaussianBlur(gray, gray, cv::Size(gaussian_kernel_size_, gaussian_kernel_size_),
                             gaussian_sigma_, gaussian_sigma_, cv::BORDER_DEFAULT);
        }

        double now = msg->header.stamp.toSec();
        if (now == 0.0) { now = ros::Time::now().toSec(); }

        if (first_image_time_ < 0.0) { first_image_time_ = now; }
        if (previous_image_time_for_dt_ >= 0.0)
        {
            const double image_dt = now - previous_image_time_for_dt_;
            if (image_dt > 0.0)
            {
                min_image_dt_ = std::min(min_image_dt_, image_dt);
                max_image_dt_ = std::max(max_image_dt_, image_dt);
            }
        }
        previous_image_time_for_dt_ = now;
        last_image_time_ = now;
        image_count_++;

        if (!have_prev_image_)
        {
            prev_gray_ = gray.clone();
            prev_time_ = now;
            cv::goodFeaturesToTrack(prev_gray_, prev_pts_, max_corners_, quality_level_, min_distance_,
                                    cv::Mat(), block_size_);
            have_prev_image_ = true;
            finish_processing_measurement();
            return;
        }

        double dt = now - prev_time_;
        if (dt <= 0.0) { dt = 1.0 / 30.0; }

        if (prev_pts_.empty() || prev_pts_.size() < 50)
        {
            cv::goodFeaturesToTrack(prev_gray_, prev_pts_, max_corners_, quality_level_, min_distance_,
                                    cv::Mat(), block_size_);
        }
        if (prev_pts_.empty())
        {
            prev_gray_ = gray.clone();
            prev_time_ = now;
            finish_processing_measurement();
            return;
        }

        std::vector<cv::Point2f> new_pts;
        std::vector<unsigned char> status;
        std::vector<float> err;
        cv::calcOpticalFlowPyrLK(prev_gray_, gray, prev_pts_, new_pts, status, err,
                                 lk_win_size_, lk_max_level_, lk_criteria_);

        if (new_pts.empty() || status.empty())
        {
            prev_gray_ = gray.clone();
            prev_time_ = now;
            prev_pts_.clear();
            finish_processing_measurement();
            return;
        }

        std::vector<cv::Point2f> good_old;
        std::vector<cv::Point2f> good_new;
        good_old.reserve(prev_pts_.size());
        good_new.reserve(new_pts.size());
        for (size_t i = 0; i < status.size(); ++i)
        {
            if (!status[i])
            {
                continue;
            }

            if (i >= err.size() ||
                err[i] > static_cast<float>(max_lk_error_))
            {
                continue;
            }

            const cv::Point2f& old_point = prev_pts_[i];
            const cv::Point2f& new_point = new_pts[i];

            if (new_point.x < 0.0f || new_point.x >= gray.cols ||
                new_point.y < 0.0f || new_point.y >= gray.rows)
            {
                continue;
            }

            const double displacement =
                cv::norm(new_point - old_point);

            if (displacement > max_flow_pixels_)
            {
                continue;
            }

            good_old.push_back(old_point);
            good_new.push_back(new_point);
        }

        tracked_features_ = static_cast<int>(good_new.size());

        double foe_x = 0.5 * static_cast<double>(gray.cols);
        double foe_y = 0.5 * static_cast<double>(gray.rows);
        dynamic_foe_valid_ =
            computeDynamicFOE(gray.size(), foe_x, foe_y);
        dynamic_foe_x_px_ = foe_x;
        dynamic_foe_y_px_ = foe_y;

        // Keep the existing CSV/visualization field name for compatibility, but
        // its meaning is now "dynamic translational FOE available".
        centre_foe_model_valid_ = dynamic_foe_valid_;

        if (dynamic_foe_valid_)
        {
            last_centre_foe_valid_time_ = now;
            have_last_centre_foe_valid_time_ = true;

            double ttc = 0.0;
            const bool ttc_valid =
                estimateTTC(
                    frame,
                    good_old,
                    good_new,
                    gray.size(),
                    dt,
                    foe_x,
                    foe_y,
                    ttc);

            if (ttc_valid && ttc < ttc_max_useful_)
            {
                ttc_miss_count_ = 0;
                if (!have_ttc_filtered_)
                {
                    ttc_filtered_ = ttc;
                    have_ttc_filtered_ = true;
                }
                else
                {
                    const double alpha = 0.2;
                    ttc_filtered_ =
                        alpha * ttc +
                        (1.0 - alpha) * ttc_filtered_;
                }
            }
            else
            {
                ttc_miss_count_++;
                if (ttc_miss_count_ > ttc_miss_limit_)
                {
                    have_ttc_filtered_ = false;
                }
            }

            // Dedicated central-corridor TTC.  Unlike the global median TTC,
            // this deliberately emphasizes the nearer part of the TTC
            // distribution in the centre of the image, where a head-on pole
            // appears.
            double frontal_ttc = 0.0;
            int frontal_ttc_candidates = 0;
            const bool frontal_ttc_valid =
                estimateFrontalTTC(
                    frame,
                    good_old,
                    good_new,
                    gray.size(),
                    dt,
                    foe_x,
                    foe_y,
                    frontal_ttc,
                    frontal_ttc_candidates);

            if (frontal_ttc_valid && frontal_ttc < ttc_max_useful_)
            {
                frontal_ttc_miss_count_ = 0;

                if (!have_frontal_ttc_filtered_)
                {
                    frontal_ttc_filtered_ = frontal_ttc;
                    have_frontal_ttc_filtered_ = true;
                }
                else
                {
                    const double alpha =
                        clamp(frontal_ttc_filter_alpha_, 0.0, 1.0);

                    frontal_ttc_filtered_ =
                        alpha * frontal_ttc +
                        (1.0 - alpha) * frontal_ttc_filtered_;
                }
            }
            else
            {
                frontal_ttc_miss_count_++;
                if (frontal_ttc_miss_count_ > frontal_ttc_miss_limit_)
                {
                    have_frontal_ttc_filtered_ = false;
                    frontal_ttc_filtered_ = 0.0;
                }
            }

            ROS_INFO_THROTTLE(
                0.35,
                "FRONT_TTC: valid=%d raw=%.3f filtered=%.3f candidates=%d danger=%d foe=(%.1f,%.1f) corridor=[%.1f,%.1f]",
                static_cast<int>(frontal_ttc_valid),
                frontal_ttc_valid ? frontal_ttc : -1.0,
                have_frontal_ttc_filtered_ ? frontal_ttc_filtered_ : -1.0,
                frontal_ttc_candidates,
                static_cast<int>(
                    have_frontal_ttc_filtered_ &&
                    frontal_ttc_filtered_ < frontal_ttc_danger_),
                dynamic_foe_x_px_,
                dynamic_foe_y_px_,
                frontal_corridor_x_min_px_,
                frontal_corridor_x_max_px_);

            vpf_risk_ =
                computeVisualPotentialField(
                    frame,
                    good_old,
                    good_new,
                    gray.size(),
                    dt,
                    foe_x,
                    foe_y);

            computeVPFRepulsionY(
                repulsive_y_,
                left_danger_,
                right_danger_);

            updateColumnGapSelection(now);

            // Smooth the raw repulsion signal here, at the source, before any
            // gain or direction-hysteresis logic ever sees it.
            repulsive_y_filtered_ =
                repulsive_alpha_ * repulsive_y_ +
                (1.0 - repulsive_alpha_) * repulsive_y_filtered_;

            // When a free gap is selected, its bearing owns the lateral side
            // preference. The legacy left/right signal is retained only as a
            // fallback when gap guidance is inactive or no safe gap exists.
            if ((!gap_guidance_active_ || gap_escape_required_) &&
                committed_gap_side_ == 0 &&
                std::abs(repulsive_y_filtered_) > 0.25)
            {
                last_avoid_direction_ =
                    sign(repulsive_y_filtered_);
            }
        }
        else
        {
            frontal_corridor_x_min_px_ = 0.0;
            frontal_corridor_x_max_px_ = 0.0;
            decayCentreFOEVisualState(now);

            ROS_INFO_THROTTLE(
                0.5,
                "DYNAMIC_FOE: unavailable; body_v=(%.2f, %.2f, %.2f) m/s, min_forward=%.2f m/s, heading_error=%.1f deg rot_active=%d",
                current_body_vx_,
                current_body_vy_,
                current_body_vz_,
                dynamic_foe_min_forward_body_speed_,
                last_heading_error_ * 180.0 / M_PI,
                static_cast<int>(rotational_apf_active_));
        }

        if (enable_visualization_)
        {
            cv::Mat vis = drawFlow(frame, good_old, good_new);
            sensor_msgs::ImagePtr vis_msg =
                cv_bridge::CvImage(msg->header, sensor_msgs::image_encodings::BGR8, vis).toImageMsg();
            vis_pub_.publish(vis_msg);
        }

        prev_gray_ = gray.clone();
        prev_time_ = now;
        cv::goodFeaturesToTrack(prev_gray_, prev_pts_, max_corners_, quality_level_, min_distance_,
                                cv::Mat(), block_size_);

        finish_processing_measurement();
    }

    void compensateYawInducedFlow(
        const cv::Point2f& point,
        const cv::Size& image_size,
        double dt,
        double raw_dx,
        double raw_dy,
        double& compensated_dx,
        double& compensated_dy) const
    {
        compensated_dx = raw_dx;
        compensated_dy = raw_dy;

        if (!enable_yaw_flow_compensation_)
        {
            return;
        }

        const double safe_dt = std::max(dt, 1e-3);

        const double fx =
            (camera_fx_pixels_ > 1.0)
                ? camera_fx_pixels_
                : 0.5 * static_cast<double>(image_size.width);

        const double fy =
            (camera_fy_pixels_ > 1.0)
                ? camera_fy_pixels_
                : fx;

        const double cx =
            0.5 * static_cast<double>(image_size.width);

        const double cy =
            0.5 * static_cast<double>(image_size.height);

        const double x_normalized =
            (static_cast<double>(point.x) - cx) / fx;

        const double y_normalized =
            (static_cast<double>(point.y) - cy) / fy;

        const double yaw_rate =
            clamp(
                current_yaw_rate_,
                -max_yaw_rate_for_compensation_,
                max_yaw_rate_for_compensation_);

        const double yaw_angle_change =
            yaw_flow_sign_ * yaw_rate * safe_dt;

        // A positive ROS body-yaw rate turns a forward-facing camera left.
        // Static scene points therefore move right in the image. These terms
        // model that rotational image displacement for a camera whose optical
        // axis is aligned with body +x, image x points right and image y down.
        const double rotational_dx =
            fx *
            (1.0 + x_normalized * x_normalized) *
            yaw_angle_change;

        const double rotational_dy =
            fy *
            x_normalized *
            y_normalized *
            yaw_angle_change;

        // Only the remaining translational component is used for looming/TTC.
        compensated_dx =
            raw_dx - rotational_dx;

        compensated_dy =
            raw_dy - rotational_dy;
    }

    cv::Mat computeVisualPotentialField(
        const cv::Mat& color_frame,
        const std::vector<cv::Point2f>& old_pts,
        const std::vector<cv::Point2f>& new_pts,
        const cv::Size& image_size,
        double dt,
        double foe_x,
        double foe_y)
    {
        const int w = image_size.width;
        const int h = image_size.height;

        cv::Mat risk_grid =
            cv::Mat::zeros(
                vpf_grid_rows_,
                vpf_grid_cols_,
                CV_32F);

        cv::Mat support_grid =
            cv::Mat::zeros(
                vpf_grid_rows_,
                vpf_grid_cols_,
                CV_32S);

        // The FOE is supplied from measured body-frame translation rather than
        // being fixed at the image centre.  After yaw-flow compensation this
        // preserves the correct radial-expansion geometry while the UAV crabs or
        // turns through dense clutter.
        const double safe_dt = std::max(dt, 1e-3);

        for (size_t i = 0; i < old_pts.size(); ++i)
        {
            const double x_old = old_pts[i].x;
            const double y_old = old_pts[i].y;
            const double x_new = new_pts[i].x;
            const double y_new = new_pts[i].y;

            if (!isTrustedObstaclePoint(
                    color_frame,
                    x_old,
                    y_old,
                    h))
            {
                continue;
            }

            const double raw_dx =
                x_new - x_old;

            const double raw_dy =
                y_new - y_old;

            double dx = 0.0;
            double dy = 0.0;

            compensateYawInducedFlow(
                old_pts[i],
                image_size,
                safe_dt,
                raw_dx,
                raw_dy,
                dx,
                dy);

            const double flow_magnitude =
                std::hypot(dx, dy);

            if (flow_magnitude < minimum_vpf_flow_pixels_)
            {
                continue;
            }

            const double rx = x_old - foe_x;
            const double ry = y_old - foe_y;
            const double radius = std::hypot(rx, ry);

            if (radius < 8.0)
            {
                continue;
            }

            const double radial_flow =
                (rx * dx + ry * dy) / radius;

            // Inward or non-expanding tracks cannot indicate looming.
            if (radial_flow <= 0.0)
            {
                continue;
            }

            // A dangerous feature should move predominantly away from the FOE,
            // not merely have a small outward component.
            const double radial_ratio =
                radial_flow / (flow_magnitude + 1e-6);

            if (radial_ratio < minimum_radial_ratio_)
            {
                continue;
            }

            const double radial_velocity =
                radial_flow / safe_dt;

            const double ttc =
                radius / radial_velocity;

            if (ttc <= 0.1 ||
                ttc >= local_danger_ttc_)
            {
                continue;
            }

            // Bounded, dimensionless risk:
            // TTC = local threshold -> 0; TTC approaches zero -> 1.
            const double ttc_risk =
                clamp(
                    (local_danger_ttc_ - ttc) /
                        local_danger_ttc_,
                    0.0,
                    1.0);

            const double risk =
                ttc_risk * radial_ratio;

            int col = static_cast<int>(
                (x_old / static_cast<double>(w)) *
                vpf_grid_cols_);

            int row = static_cast<int>(
                (y_old / static_cast<double>(h)) *
                vpf_grid_rows_);

            col = std::max(
                0,
                std::min(vpf_grid_cols_ - 1, col));

            row = std::max(
                0,
                std::min(vpf_grid_rows_ - 1, row));

            risk_grid.at<float>(row, col) +=
                static_cast<float>(risk);

            support_grid.at<int>(row, col)++;
        }

        // Average risk within each cell. This prevents highly textured image
        // regions from appearing more dangerous merely because they contain
        // more Shi-Tomasi features.
        //
        // After averaging, reject weak cells using a fixed absolute threshold.
        // This prevents low-level floor/background responses from contributing
        // to left/right danger and the lateral avoidance command.
        const double minimum_cell_risk = 0.15;

        for (int row = 0; row < risk_grid.rows; ++row)
        {
            for (int col = 0; col < risk_grid.cols; ++col)
            {
                const int support =
                    support_grid.at<int>(row, col);

                float& cell_risk =
                    risk_grid.at<float>(row, col);

                if (support >= minimum_cell_support_)
                {
                    cell_risk /=
                        static_cast<float>(support);

                    if (cell_risk < minimum_cell_risk)
                    {
                        cell_risk = 0.0f;
                    }
                }
                else
                {
                    cell_risk = 0.0f;
                }
            }
        }

        // No per-frame maximum normalization. Every cell remains on the same
        // absolute [0, 1] danger scale from frame to frame.
        return risk_grid;
    }

    void computeVPFRepulsionY(
        double& repulsive_y,
        double& left_danger,
        double& right_danger)
    {
        left_danger = 0.0;
        right_danger = 0.0;
        repulsive_y = 0.0;

        int active_left_cells = 0;
        int active_right_cells = 0;

        for (int row = 0; row < vpf_risk_.rows; ++row)
        {
            for (int col = 0; col < vpf_risk_.cols; ++col)
            {
                const double risk =
                    static_cast<double>(
                        vpf_risk_.at<float>(row, col));

                if (risk <= 0.0)
                {
                    continue;
                }

                if (col < vpf_grid_cols_ / 2)
                {
                    left_danger += risk;
                    active_left_cells++;
                }
                else
                {
                    right_danger += risk;
                    active_right_cells++;
                }
            }
        }

        if (active_left_cells > 0)
        {
            left_danger /=
                static_cast<double>(active_left_cells);
        }

        if (active_right_cells > 0)
        {
            right_danger /=
                static_cast<double>(active_right_cells);
        }

        // With the standard ROS body convention, positive lateral velocity is
        // body-left. Therefore image-right danger produces positive repulsion.
        repulsive_y =
            right_danger - left_danger;
    }



    void resetGapGuidanceState()
    {
        column_risk_.assign(
            std::max(1, vpf_grid_cols_),
            0.0);

        gap_guidance_active_ = false;
        safe_gap_available_ = false;
        gap_escape_required_ = false;
        selected_gap_start_col_ = -1;
        selected_gap_end_col_ = -1;
        selected_gap_center_col_ = -1.0;
        selected_gap_width_cols_ = 0;
        selected_gap_heading_offset_rad_ = 0.0;
        selected_gap_mean_risk_ = 0.0;
        selected_gap_score_ = std::numeric_limits<double>::max();
        max_column_risk_ = 0.0;
        last_gap_selection_time_ = 0.0;
        last_gap_obstacle_time_ = 0.0;
        have_last_gap_obstacle_time_ = false;

        // Persistent corridor-side commitment. This is deliberately separate
        // from selected_gap_* because selected_gap_* is cleared when no safe
        // corridor exists for a frame.
        committed_gap_side_ = 0;
        last_committed_gap_time_ = 0.0;

        // Recent no-safe-gap latch used by rotational APF activation.
        last_no_safe_gap_time_ = 0.0;
        have_last_no_safe_gap_time_ = false;
    }

    // Collapse the 16x8 VPF only vertically: one danger value is retained for
    // every image column. Horizontal structure is therefore preserved for gap
    // selection. For each column, max risk gives timely response to a pole while
    // mean risk prevents one isolated cell from dominating the whole column.
    void computeColumnRiskProfile()
    {
        column_risk_.assign(
            std::max(1, vpf_grid_cols_),
            0.0);

        max_column_risk_ = 0.0;

        if (vpf_risk_.empty() ||
            vpf_risk_.cols != vpf_grid_cols_ ||
            vpf_risk_.rows != vpf_grid_rows_)
        {
            return;
        }

        for (int col = 0; col < vpf_grid_cols_; ++col)
        {
            double risk_sum = 0.0;
            double risk_max = 0.0;
            int trusted_rows = 0;

            for (int row = 0; row < vpf_grid_rows_; ++row)
            {
                const double row_center_fraction =
                    (static_cast<double>(row) + 0.5) /
                    static_cast<double>(vpf_grid_rows_);

                if (row_center_fraction < floor_row_min_frac_ ||
                    row_center_fraction > floor_row_max_frac_)
                {
                    continue;
                }

                const double risk =
                    clamp(
                        static_cast<double>(
                            vpf_risk_.at<float>(row, col)),
                        0.0,
                        1.0);

                risk_sum += risk;
                risk_max = std::max(risk_max, risk);
                trusted_rows++;
            }

            const double mean_risk =
                (trusted_rows > 0)
                    ? risk_sum / static_cast<double>(trusted_rows)
                    : 0.0;

            const double column_risk =
                clamp(
                    0.65 * risk_max +
                    0.35 * mean_risk,
                    0.0,
                    1.0);

            column_risk_[col] = column_risk;
            max_column_risk_ =
                std::max(max_column_risk_, column_risk);
        }

        // ------------------------------------------------------
        // Fuse the independent frontal TTC safety channel into
        // the SAME column map used by gap selection.
        //
        // Previously frontal TTC could report an imminent obstacle
        // while sparse VPF cells in the same region were zero, so
        // the gap selector could incorrectly call that region free.
        // ------------------------------------------------------
        if (have_frontal_ttc_filtered_ &&
            frontal_ttc_filtered_ < frontal_ttc_danger_)
        {
            const double frontal_risk =
                clamp(
                    1.0 -
                        frontal_ttc_filtered_ /
                        std::max(0.1, frontal_ttc_danger_),
                    0.0,
                    1.0);

            // A frontal TTC warning is hard safety evidence for gap selection.
            // Even near the TTC danger horizon, these columns must not fall
            // below the "safe gap" threshold and be interpreted as free space.
            const double blocking_risk =
                std::max(
                    frontal_risk,
                    gap_safe_column_risk_threshold_ + 0.01);

            const double image_width =
                static_cast<double>(
                    std::max(1, last_image_width_));

            int start_col =
                static_cast<int>(
                    frontal_corridor_x_min_px_ /
                    image_width *
                    static_cast<double>(vpf_grid_cols_));

            int end_col =
                static_cast<int>(
                    frontal_corridor_x_max_px_ /
                    image_width *
                    static_cast<double>(vpf_grid_cols_));

            start_col =
                std::max(
                    0,
                    std::min(vpf_grid_cols_ - 1, start_col));

            end_col =
                std::max(
                    0,
                    std::min(vpf_grid_cols_ - 1, end_col));

            if (start_col > end_col)
            {
                std::swap(start_col, end_col);
            }

            for (int col = start_col; col <= end_col; ++col)
            {
                column_risk_[col] =
                    std::max(
                        column_risk_[col],
                        blocking_risk);

                max_column_risk_ =
                    std::max(
                        max_column_risk_,
                        column_risk_[col]);
            }
        }

        // ------------------------------------------------------
        // Sparse LK features tend to lie on obstacle edges. Give
        // every dangerous column ONE-column spatial support so a
        // pole interior is not interpreted as a narrow free gap.
        // This deliberately stays simple: no TTC-derived range,
        // no variable inflation radius, and no extra tuning knobs.
        // ------------------------------------------------------
        const std::vector<double> pre_expansion_risk =
            column_risk_;

        for (int col = 0; col < vpf_grid_cols_; ++col)
        {
            if (pre_expansion_risk[col] <=
                gap_safe_column_risk_threshold_)
            {
                continue;
            }

            if (col > 0)
            {
                column_risk_[col - 1] =
                    std::max(
                        column_risk_[col - 1],
                        pre_expansion_risk[col]);
            }

            if (col + 1 < vpf_grid_cols_)
            {
                column_risk_[col + 1] =
                    std::max(
                        column_risk_[col + 1],
                        pre_expansion_risk[col]);
            }
        }

        // Close only tiny INTERNAL safe holes. Sparse LK often gives strong
        // features on the two edges of a pole but little texture in its centre.
        // A 1-2 column zero-risk island between dangerous columns is therefore
        // treated as missing visual support, not as a flyable corridor.
        // Wider gaps and gaps touching an image edge remain untouched.
        int scan_col = 0;
        while (scan_col < vpf_grid_cols_)
        {
            if (column_risk_[scan_col] > gap_safe_column_risk_threshold_)
            {
                ++scan_col;
                continue;
            }

            const int hole_start = scan_col;
            while (scan_col < vpf_grid_cols_ &&
                   column_risk_[scan_col] <= gap_safe_column_risk_threshold_)
            {
                ++scan_col;
            }

            const int hole_end = scan_col - 1;
            const int hole_width = hole_end - hole_start + 1;
            const bool bounded_left =
                hole_start > 0 &&
                column_risk_[hole_start - 1] > gap_safe_column_risk_threshold_;
            const bool bounded_right =
                scan_col < vpf_grid_cols_ &&
                column_risk_[scan_col] > gap_safe_column_risk_threshold_;

            if (bounded_left && bounded_right && hole_width <= 2)
            {
                const double fill_risk =
                    std::max(
                        gap_safe_column_risk_threshold_ + 0.01,
                        std::min(
                            column_risk_[hole_start - 1],
                            column_risk_[scan_col]));

                for (int fill_col = hole_start; fill_col <= hole_end; ++fill_col)
                {
                    column_risk_[fill_col] =
                        std::max(column_risk_[fill_col], fill_risk);
                }
            }
        }

        max_column_risk_ = 0.0;
        for (int col = 0; col < vpf_grid_cols_; ++col)
        {
            max_column_risk_ =
                std::max(
                    max_column_risk_,
                    column_risk_[col]);
        }
    }

    double gapColumnToHeadingOffsetRad(double column_center) const
    {
        const double image_width =
            std::max(1, last_image_width_);

        const double cx =
            0.5 * static_cast<double>(image_width);

        const double fx =
            (camera_fx_pixels_ > 1.0)
                ? camera_fx_pixels_
                : cx;

        const double u =
            (column_center + 0.5) /
            static_cast<double>(std::max(1, vpf_grid_cols_)) *
            static_cast<double>(image_width);

        // Positive image x is camera/body-right. Positive ROS yaw/body-y is
        // left, therefore the steering-yaw offset has the opposite sign.
        return
            -std::atan2(
                u - cx,
                fx);
    }

    // Select the safest contiguous low-risk horizontal corridor. Candidate gaps
    // are scored by their mean risk, angular distance from the goal direction,
    // continuity with the previous selected gap, and width. The risk term is
    // dominant; the other terms only break ties between similarly safe gaps.
    void updateColumnGapSelection(double now)
    {
        if (!enable_gap_guidance_)
        {
            gap_guidance_active_ = false;
            safe_gap_available_ = false;
            gap_escape_required_ = false;
            return;
        }

        computeColumnRiskProfile();

        const bool obstacle_field_active =
            max_column_risk_ >= gap_activation_risk_threshold_;

        if (obstacle_field_active)
        {
            last_gap_obstacle_time_ = now;
            have_last_gap_obstacle_time_ = true;
        }

        const bool recently_active =
            have_last_gap_obstacle_time_ &&
            (now - last_gap_obstacle_time_) <=
                gap_release_delay_s_;

        if (!obstacle_field_active)
        {
            if (recently_active)
            {
                // No fresh obstacle field is currently available (commonly while
                // the centre-FOE gate is invalid during a turn). Preserve the
                // previously selected gap/no-gap decision for the release delay
                // instead of interpreting the decayed all-zero VPF as a new
                // full-width free corridor.
                if (safe_gap_available_)
                {
                    gap_guidance_active_ = true;
                }
                return;
            }

            gap_guidance_active_ = false;
            safe_gap_available_ = false;
            gap_escape_required_ = false;
            selected_gap_start_col_ = -1;
            selected_gap_end_col_ = -1;
            selected_gap_center_col_ = -1.0;
            selected_gap_width_cols_ = 0;
            selected_gap_heading_offset_rad_ = 0.0;
            selected_gap_mean_risk_ = 0.0;
            selected_gap_score_ = std::numeric_limits<double>::max();

            // The obstacle episode has genuinely ended, so a future episode
            // may choose either side again.
            committed_gap_side_ = 0;
            last_committed_gap_time_ = 0.0;
            have_last_no_safe_gap_time_ = false;
            last_no_safe_gap_time_ = 0.0;
            return;
        }

        struct GapCandidate
        {
            int start_col;
            int end_col;
            int width;
            double center_col;
            double mean_risk;
            double heading_offset;
            double score;
        };

        std::vector<GapCandidate> candidates;

        int col = 0;
        while (col < vpf_grid_cols_)
        {
            if (column_risk_[col] > gap_safe_column_risk_threshold_)
            {
                col++;
                continue;
            }

            const int start = col;
            double risk_sum = 0.0;

            while (col < vpf_grid_cols_ &&
                   column_risk_[col] <= gap_safe_column_risk_threshold_)
            {
                risk_sum += column_risk_[col];
                col++;
            }

            const int end = col - 1;
            const int width = end - start + 1;

            if (width < std::max(1, gap_min_width_cols_))
            {
                continue;
            }

            GapCandidate candidate;
            candidate.start_col = start;
            candidate.end_col = end;
            candidate.width = width;
            candidate.center_col =
                0.5 * static_cast<double>(start + end);
            candidate.mean_risk =
                risk_sum / static_cast<double>(width);
            candidate.heading_offset =
                gapColumnToHeadingOffsetRad(
                    candidate.center_col);

            const double goal_heading_world =
                std::atan2(
                    goal_y_ - current_pose_.position.y,
                    goal_x_ - current_pose_.position.x);

            const double goal_heading_offset =
                normalizeAngle(
                    goal_heading_world - current_yaw_);

            const double half_fov =
                std::atan2(
                    0.5 * static_cast<double>(
                        std::max(1, last_image_width_)),
                    std::max(1.0, camera_fx_pixels_));

            const double angular_normalizer =
                std::max(0.10, half_fov);

            const double goal_cost =
                std::abs(
                    normalizeAngle(
                        candidate.heading_offset -
                        goal_heading_offset)) /
                angular_normalizer;

            double continuity_cost = 0.0;
            if (selected_gap_center_col_ >= 0.0)
            {
                continuity_cost =
                    std::abs(
                        normalizeAngle(
                            candidate.heading_offset -
                            selected_gap_heading_offset_rad_)) /
                    angular_normalizer;
            }

            const double width_fraction =
                static_cast<double>(candidate.width) /
                static_cast<double>(
                    std::max(1, vpf_grid_cols_));

            candidate.score =
                candidate.mean_risk +
                gap_goal_bias_weight_ * goal_cost +
                gap_continuity_weight_ * continuity_cost -
                gap_width_bonus_weight_ * width_fraction;

            candidates.push_back(candidate);
        }

        // Keep the selected corridor while it remains part of a currently safe
        // candidate. Refreshing the commitment time prevents frame-to-frame
        // optical-flow noise from causing left/right gap switching.
        if (selected_gap_center_col_ >= 0.0 &&
            (now - last_gap_selection_time_) < gap_hold_time_s_)
        {
            for (size_t i = 0; i < candidates.size(); ++i)
            {
                const GapCandidate& candidate = candidates[i];
                if (selected_gap_center_col_ >= candidate.start_col &&
                    selected_gap_center_col_ <= candidate.end_col)
                {
                    safe_gap_available_ = true;
                    gap_guidance_active_ = true;
                    gap_escape_required_ = false;
                    selected_gap_start_col_ = candidate.start_col;
                    selected_gap_end_col_ = candidate.end_col;
                    selected_gap_center_col_ = candidate.center_col;
                    selected_gap_width_cols_ = candidate.width;
                    selected_gap_heading_offset_rad_ =
                        candidate.heading_offset;
                    selected_gap_mean_risk_ = candidate.mean_risk;
                    selected_gap_score_ = candidate.score;

                    // Renew the commitment while the current corridor is still
                    // safe. Without this refresh, the hold interval would expire
                    // and the selector could jump to a different gap even though
                    // this corridor remains valid.
                    last_gap_selection_time_ = now;

                    if (std::abs(selected_gap_heading_offset_rad_) > 0.03)
                    {
                        const int selected_side =
                            static_cast<int>(
                                sign(selected_gap_heading_offset_rad_));

                        last_avoid_direction_ = selected_side;
                        committed_gap_side_ = selected_side;
                        last_committed_gap_time_ = now;
                    }
                    return;
                }
            }
        }

        if (!candidates.empty())
        {
            // --------------------------------------------------------------
            // Same-side gap hysteresis.
            // --------------------------------------------------------------
            // Sparse optical-flow support can make the best numerical gap jump
            // from image-left to image-right from one frame to the next.  Once
            // a meaningful side has been selected, keep choosing among safe
            // candidates on that SAME side while at least one still exists.
            // Only switch sides when the committed side genuinely has no valid
            // candidate left.  This is stronger than the ordinary continuity
            // score and prevents rapid left/right steering reversals.
            // Use a persistent side commitment rather than deriving the side
            // from selected_gap_*. The selected gap is intentionally cleared
            // during a NO SAFE GAP frame, but the side commitment must survive
            // that temporary loss of visual corridor information.
            const int previous_gap_side =
                committed_gap_side_;

            bool same_side_candidate_exists = false;
            if (previous_gap_side != 0)
            {
                for (size_t i = 0; i < candidates.size(); ++i)
                {
                    if (std::abs(candidates[i].heading_offset) <= 0.03)
                    {
                        continue;
                    }

                    const int candidate_side =
                        static_cast<int>(
                            sign(candidates[i].heading_offset));

                    if (candidate_side == previous_gap_side)
                    {
                        same_side_candidate_exists = true;
                        break;
                    }
                }
            }

            size_t best_index = 0;
            bool best_index_set = false;

            for (size_t i = 0; i < candidates.size(); ++i)
            {
                if (same_side_candidate_exists)
                {
                    if (std::abs(candidates[i].heading_offset) <= 0.03)
                    {
                        continue;
                    }

                    const int candidate_side =
                        static_cast<int>(
                            sign(candidates[i].heading_offset));

                    if (candidate_side != previous_gap_side)
                    {
                        continue;
                    }
                }

                if (!best_index_set ||
                    candidates[i].score < candidates[best_index].score)
                {
                    best_index = i;
                    best_index_set = true;
                }
            }

            // Defensive fallback: this should only be needed if all candidates
            // were exactly inside the small centre deadband.
            if (!best_index_set)
            {
                best_index = 0;
                for (size_t i = 1; i < candidates.size(); ++i)
                {
                    if (candidates[i].score < candidates[best_index].score)
                    {
                        best_index = i;
                    }
                }
            }

            const GapCandidate& best =
                candidates[best_index];

            safe_gap_available_ = true;
            gap_guidance_active_ = true;
            gap_escape_required_ = false;
            selected_gap_start_col_ = best.start_col;
            selected_gap_end_col_ = best.end_col;
            selected_gap_center_col_ = best.center_col;
            selected_gap_width_cols_ = best.width;
            selected_gap_heading_offset_rad_ = best.heading_offset;
            selected_gap_mean_risk_ = best.mean_risk;
            selected_gap_score_ = best.score;
            last_gap_selection_time_ = now;

            if (std::abs(selected_gap_heading_offset_rad_) > 0.03)
            {
                const int selected_side =
                    static_cast<int>(
                        sign(selected_gap_heading_offset_rad_));

                last_avoid_direction_ = selected_side;
                committed_gap_side_ = selected_side;
                last_committed_gap_time_ = now;
            }

            ROS_INFO_THROTTLE(
                0.35,
                "GAP_SELECT: cols=%d..%d width=%d center=%.2f heading_offset=%.1f deg mean_risk=%.3f max_col_risk=%.3f score=%.3f",
                selected_gap_start_col_,
                selected_gap_end_col_,
                selected_gap_width_cols_,
                selected_gap_center_col_,
                selected_gap_heading_offset_rad_ * 180.0 / M_PI,
                selected_gap_mean_risk_,
                max_column_risk_,
                selected_gap_score_);
            return;
        }

        // Obstacle risk is present but no sufficiently wide low-risk corridor
        // exists. Keep the legacy side preference as a tie-breaker, then request
        // the rotational APF. Its existing start delay suppresses one-frame noise.
        safe_gap_available_ = false;
        gap_guidance_active_ = false;
        gap_escape_required_ = true;

        // Latch this evidence for a short window. Sparse LK can make a safe gap
        // reappear for one or two frames; without the latch, the rotational APF
        // start delay is continually reset and never activates.
        last_no_safe_gap_time_ = now;
        have_last_no_safe_gap_time_ = true;

        selected_gap_start_col_ = -1;
        selected_gap_end_col_ = -1;
        selected_gap_center_col_ = -1.0;
        selected_gap_width_cols_ = 0;
        selected_gap_heading_offset_rad_ = 0.0;
        selected_gap_mean_risk_ = 1.0;
        selected_gap_score_ = std::numeric_limits<double>::max();

        const double legacy_side_preference =
            right_danger_ - left_danger_;

        // A NO-SAFE-GAP frame must NOT reverse a side that was already
        // committed by a valid corridor. The left/right VPF difference is noisy
        // exactly in this situation and was causing RIGHT -> LEFT -> RIGHT
        // fallback flips. Only use it when no corridor side has been committed.
        if (committed_gap_side_ != 0)
        {
            last_avoid_direction_ =
                static_cast<double>(committed_gap_side_);
        }
        else if (std::abs(legacy_side_preference) > 0.05)
        {
            last_avoid_direction_ =
                sign(legacy_side_preference);
        }

        ROS_WARN_THROTTLE(
            0.50,
            "GAP_SELECT: NO SAFE GAP (max column risk=%.3f). Requesting rotational APF; fallback side=%s.",
            max_column_risk_,
            last_avoid_direction_ >= 0.0 ? "LEFT" : "RIGHT");
    }


    void updateProgressDetector(double distance_to_goal, bool danger_active)
    {
        const double now = ros::Time::now().toSec();

        if (!have_prev_distance_to_goal_)
        {
            prev_distance_to_goal_ = distance_to_goal;
            progress_timer_start_ = now;
            have_prev_distance_to_goal_ = true;
            have_progress_timer_start_ = true;
            progress_poor_ = false;
            progress_rate_ = 0.0;
            return;
        }
        if (!have_progress_timer_start_)
        {
            progress_timer_start_ = now;
            prev_distance_to_goal_ = distance_to_goal;
            have_progress_timer_start_ = true;
            progress_poor_ = false;
            progress_rate_ = 0.0;
            return;
        }

        const double dt_progress = now - progress_timer_start_;
        if (dt_progress >= progress_window_)
        {
            progress_rate_ = (prev_distance_to_goal_ - distance_to_goal) / dt_progress;
            progress_poor_ = danger_active && (progress_rate_ < min_progress_rate_);
            prev_distance_to_goal_ = distance_to_goal;
            progress_timer_start_ = now;
        }
    }

    // Return the signed lateral coordinate relative to the straight
    // start-to-goal path. For the current horizontal reference path this is
    // equivalent to a y offset, but it also remains valid when the path has a
    // non-zero heading in the world frame.
    double lateralCoordinateRelativeToPath(
        double x,
        double y) const
    {
        const double path_dx = goal_x_ - start_x_;
        const double path_dy = goal_y_ - start_y_;
        const double path_length = std::hypot(path_dx, path_dy);

        if (path_length <= 1e-6)
        {
            return y - start_y_;
        }

        return
            (-path_dy * (x - start_x_) +
              path_dx * (y - start_y_)) /
            path_length;
    }

    void clearOscillationEpisode(bool reset_command_sign = true)
    {
        oscillation_count_ = 0;
        oscillation_active_ = false;
        have_oscillation_episode_start_ = false;
        have_last_oscillation_time_ = false;
        have_last_accepted_reversal_lateral_ = false;

        oscillation_episode_start_time_ = 0.0;
        last_oscillation_time_ = 0.0;
        oscillation_episode_start_x_ = 0.0;
        oscillation_episode_start_y_ = 0.0;
        oscillation_episode_start_distance_to_goal_ = 0.0;
        last_accepted_reversal_lateral_ = 0.0;

        if (reset_command_sign)
        {
            prev_vy_sign_ = 0.0;
        }
    }

    // Physical, progress-aware oscillation detector.
    //
    // A commanded lateral sign change is accepted as one reversal only when
    // the UAV has moved a meaningful cross-track distance since the previous
    // accepted reversal. Persistent failure additionally requires:
    //   - enough accepted reversals,
    //   - poor progress toward the goal,
    //   - continued obstacle danger, and
    //   - confinement to one local region.
    //
    // This prevents legitimate left/right obstacle avoidance at different
    // locations in a cluttered environment from being marked as oscillation.
    void updateOscillationDetector(
        double vy_command,
        double x_position,
        double y_position,
        double distance_to_goal,
        bool danger_active)
    {
        const double now = ros::Time::now().toSec();

        // Once obstacle danger has disappeared, the previous local episode is
        // no longer relevant to the current avoidance situation.
        if (!danger_active)
        {
            clearOscillationEpisode();
            return;
        }

        double current_sign = 0.0;
        if (vy_command >= vy_sign_threshold_)
        {
            current_sign = 1.0;
        }
        else if (vy_command <= -vy_sign_threshold_)
        {
            current_sign = -1.0;
        }

        // Ignore small lateral commands around zero. Clear a stale episode if
        // there has not been a meaningful reversal for the configured memory.
        if (current_sign == 0.0)
        {
            if (have_last_oscillation_time_ &&
                (now - last_oscillation_time_) >
                    oscillation_memory_time_)
            {
                ROS_INFO(
                    "Oscillation episode cleared after %.2f s "
                    "without a meaningful physical reversal.",
                    now - last_oscillation_time_);

                clearOscillationEpisode();
            }
            return;
        }

        const double lateral_coordinate =
            lateralCoordinateRelativeToPath(
                x_position,
                y_position);

        if (prev_vy_sign_ == 0.0)
        {
            prev_vy_sign_ = current_sign;
            last_accepted_reversal_lateral_ = lateral_coordinate;
            have_last_accepted_reversal_lateral_ = true;
            return;
        }

        if (current_sign != prev_vy_sign_)
        {
            const double lateral_excursion =
                have_last_accepted_reversal_lateral_
                    ? std::abs(
                          lateral_coordinate -
                          last_accepted_reversal_lateral_)
                    : std::numeric_limits<double>::infinity();

            // Reject command chatter. The actual UAV must first cross a
            // meaningful lateral distance before the sign change is counted.
            if (lateral_excursion >=
                minimum_lateral_reversal_distance_)
            {
                oscillation_count_++;
                total_oscillation_count_++;
                last_oscillation_time_ = now;
                have_last_oscillation_time_ = true;
                last_accepted_reversal_lateral_ = lateral_coordinate;
                have_last_accepted_reversal_lateral_ = true;

                if (!have_oscillation_episode_start_)
                {
                    oscillation_episode_start_time_ = now;
                    oscillation_episode_start_x_ = x_position;
                    oscillation_episode_start_y_ = y_position;
                    oscillation_episode_start_distance_to_goal_ =
                        distance_to_goal;
                    have_oscillation_episode_start_ = true;

                    ROS_WARN(
                        "Oscillation episode started: x=%.2f, y=%.2f, "
                        "lateral excursion=%.2f m.",
                        x_position,
                        y_position,
                        lateral_excursion);
                }

                if (oscillation_count_ >= oscillation_limit_)
                {
                    oscillation_active_ = true;
                }

                ROS_WARN_THROTTLE(
                    0.5,
                    "Meaningful lateral reversal: count=%d, "
                    "x=%.2f, y=%.2f, excursion=%.2f m.",
                    oscillation_count_,
                    x_position,
                    y_position,
                    lateral_excursion);
            }

            // Always consume the sign transition. Otherwise one noisy crossing
            // could be reconsidered on every control tick.
            prev_vy_sign_ = current_sign;
        }

        if (!have_oscillation_episode_start_)
        {
            return;
        }

        const double displacement_from_episode_start =
            std::hypot(
                x_position - oscillation_episode_start_x_,
                y_position - oscillation_episode_start_y_);

        const double goal_progress =
            oscillation_episode_start_distance_to_goal_ -
            distance_to_goal;

        // Turns made at substantially different locations are navigation, not
        // local oscillation. Clear the episode after useful spatial or goal
        // progress, even if the lateral command later changes sign again.
        if (displacement_from_episode_start >
                oscillation_trap_radius_ ||
            goal_progress >=
                oscillation_minimum_goal_progress_)
        {
            ROS_INFO(
                "Oscillation episode cleared after useful progress: "
                "displacement=%.2f m, goal progress=%.2f m.",
                displacement_from_episode_start,
                goal_progress);

            clearOscillationEpisode(false);
            prev_vy_sign_ = current_sign;
            last_accepted_reversal_lateral_ = lateral_coordinate;
            have_last_accepted_reversal_lateral_ = true;
            return;
        }

        if (have_last_oscillation_time_ &&
            (now - last_oscillation_time_) >
                oscillation_memory_time_)
        {
            ROS_INFO(
                "Oscillation episode cleared after %.2f s "
                "without a meaningful physical reversal.",
                now - last_oscillation_time_);

            clearOscillationEpisode(false);
            prev_vy_sign_ = current_sign;
            last_accepted_reversal_lateral_ = lateral_coordinate;
            have_last_accepted_reversal_lateral_ = true;
            return;
        }

        if (oscillation_active_ &&
            have_last_oscillation_time_)
        {
            const double episode_duration =
                now - oscillation_episode_start_time_;

            const double time_since_last_reversal =
                now - last_oscillation_time_;

            const bool reversals_are_still_recent =
                time_since_last_reversal <=
                oscillation_memory_time_;

            const bool spatially_trapped =
                displacement_from_episode_start <=
                oscillation_trap_radius_;

            const bool insufficient_goal_progress =
                goal_progress <
                oscillation_minimum_goal_progress_;

            const bool genuinely_stuck =
                progress_poor_ &&
                danger_active &&
                spatially_trapped &&
                insufficient_goal_progress;

            if (episode_duration >=
                    oscillation_failure_time_ &&
                reversals_are_still_recent &&
                genuinely_stuck)
            {
                oscillation_failure_ = true;

                ROS_ERROR(
                    "Persistent physical oscillation detected: "
                    "duration=%.2f s, count=%d, total=%d, "
                    "displacement=%.2f m, goal_progress=%.2f m, "
                    "progress_rate=%.3f m/s.",
                    episode_duration,
                    oscillation_count_,
                    total_oscillation_count_,
                    displacement_from_episode_start,
                    goal_progress,
                    progress_rate_);
            }
        }
    }

    // Estimate TTC only in the central image corridor.  This channel is
    // intentionally separate from the global TTC median: in dense clutter the
    // global median can be dominated by many farther features while a small
    // pole directly ahead is the actual collision threat.
    bool estimateFrontalTTC(
        const cv::Mat& color_frame,
        const std::vector<cv::Point2f>& old_pts,
        const std::vector<cv::Point2f>& new_pts,
        const cv::Size& image_size,
        double dt,
        double foe_x,
        double foe_y,
        double& ttc_out,
        int& candidate_count_out)
    {
        const int w = image_size.width;
        const int h = image_size.height;
        const double safe_dt = std::max(dt, 1e-3);

        // Keep the original corridor width (default 25% of the image), but move
        // its centre to the measured translational FOE.  The corridor therefore
        // follows the actual direction of travel rather than the optical-axis
        // centre while the UAV has lateral velocity.
        const double min_frac =
            clamp(frontal_ttc_min_x_fraction_, 0.0, 1.0);
        const double max_frac =
            clamp(frontal_ttc_max_x_fraction_, min_frac, 1.0);
        const double corridor_width =
            std::max(1.0,
                     (max_frac - min_frac) * static_cast<double>(w));
        const double half_width = 0.5 * corridor_width;

        const double x_min =
            clamp(foe_x - half_width, 0.0, static_cast<double>(w));
        const double x_max =
            clamp(foe_x + half_width, 0.0, static_cast<double>(w));

        frontal_corridor_x_min_px_ = x_min;
        frontal_corridor_x_max_px_ = x_max;
        candidate_count_out = 0;

        if (x_max - x_min < 1.0)
        {
            return false;
        }

        std::vector<double> taus;
        taus.reserve(old_pts.size());

        for (size_t i = 0; i < old_pts.size(); ++i)
        {
            const double x_old = old_pts[i].x;
            const double y_old = old_pts[i].y;

            if (x_old < x_min || x_old >= x_max)
            {
                continue;
            }

            if (!isTrustedObstaclePoint(
                    color_frame, x_old, y_old, h))
            {
                continue;
            }

            const double rx = x_old - foe_x;
            const double ry = y_old - foe_y;

            const double raw_flow_x =
                new_pts[i].x - old_pts[i].x;
            const double raw_flow_y =
                new_pts[i].y - old_pts[i].y;

            double flow_x = 0.0;
            double flow_y = 0.0;

            compensateYawInducedFlow(
                old_pts[i],
                image_size,
                safe_dt,
                raw_flow_x,
                raw_flow_y,
                flow_x,
                flow_y);

            const double r = std::hypot(rx, ry);

            // Keep only a very small singular region around the exact FOE.
            if (r < 8.0)
            {
                continue;
            }

            const double radial_flow =
                (rx * flow_x + ry * flow_y) / r;
            const double radial_velocity =
                radial_flow / safe_dt;

            if (radial_velocity <= 1e-3)
            {
                continue;
            }

            const double tau = r / radial_velocity;
            if (tau > 0.1 && tau < 20.0)
            {
                taus.push_back(tau);
            }
        }

        candidate_count_out = static_cast<int>(taus.size());

        if (candidate_count_out <
            std::max(1, frontal_ttc_min_features_))
        {
            return false;
        }

        std::sort(taus.begin(), taus.end());

        // A lower quantile is robust to a single bad minimum but still gives
        // priority to the nearest coherent looming structure.
        const double q =
            clamp(frontal_ttc_quantile_, 0.0, 1.0);
        const double index =
            q * static_cast<double>(taus.size() - 1);
        const size_t lower =
            static_cast<size_t>(std::floor(index));
        const size_t upper =
            static_cast<size_t>(std::ceil(index));
        const double fraction =
            index - static_cast<double>(lower);

        ttc_out =
            (1.0 - fraction) * taus[lower] +
            fraction * taus[upper];

        return std::isfinite(ttc_out);
    }

    bool estimateTTC(const cv::Mat& color_frame,
                     const std::vector<cv::Point2f>& old_pts,
                     const std::vector<cv::Point2f>& new_pts,
                     const cv::Size& image_size,
                     double dt,
                     double foe_x,
                     double foe_y,
                     double& ttc_out)
    {
        const int w = image_size.width;
        const int h = image_size.height;
        const double safe_dt = std::max(dt, 1e-3);

        std::vector<double> taus;
        taus.reserve(old_pts.size());

        for (size_t i = 0; i < old_pts.size(); ++i)
        {
            const double x_old = old_pts[i].x;
            const double y_old = old_pts[i].y;
            const double x_new = new_pts[i].x;
            const double y_new = new_pts[i].y;

            if (!isTrustedObstaclePoint(color_frame, x_old, y_old, h)) { continue; }

            const double rx = x_old - foe_x;
            const double ry = y_old - foe_y;

            const double raw_flow_x =
                x_new - x_old;

            const double raw_flow_y =
                y_new - y_old;

            double flow_x = 0.0;
            double flow_y = 0.0;

            compensateYawInducedFlow(
                old_pts[i],
                image_size,
                safe_dt,
                raw_flow_x,
                raw_flow_y,
                flow_x,
                flow_y);

            const double r =
                std::sqrt(rx * rx + ry * ry);

            if (r < 8.0)
            {
                continue;
            }

            const double radial_flow =
                (rx * flow_x + ry * flow_y) / r;

            const double radial_velocity =
                radial_flow / safe_dt;

            if (radial_velocity > 1e-3)
            {
                const double tau = r / radial_velocity;
                if (tau > 0.1 && tau < 20.0) { taus.push_back(tau); }
            }
        }

        if (taus.size() < 5) { return false; }

        std::sort(taus.begin(), taus.end());
        const size_t mid = taus.size() / 2;
        ttc_out = (taus.size() % 2 == 0) ? 0.5 * (taus[mid - 1] + taus[mid]) : taus[mid];
        return true;
    }

    cv::Mat drawFlow(const cv::Mat& frame,
                     const std::vector<cv::Point2f>& old_pts,
                     const std::vector<cv::Point2f>& new_pts)
    {
        cv::Mat output = frame.clone();

        for (size_t i = 0; i < old_pts.size(); ++i)
        {
            cv::arrowedLine(output,
                            cv::Point(static_cast<int>(old_pts[i].x), static_cast<int>(old_pts[i].y)),
                            cv::Point(static_cast<int>(new_pts[i].x), static_cast<int>(new_pts[i].y)),
                            cv::Scalar(0, 255, 0), 2, cv::LINE_8, 0, 0.3);
            cv::circle(output, cv::Point(static_cast<int>(new_pts[i].x), static_cast<int>(new_pts[i].y)),
                      3, cv::Scalar(0, 0, 255), -1);
        }

        // --------------------------------------------------------------
        // Visualise the complete 16 x 8 VPF as an image-space heat map.
        //
        // Blue   = low risk
        // Green  = moderate risk
        // Yellow = high risk
        // Red    = very high risk
        //
        // This block is diagnostic only. It does not alter vpf_risk_ or any
        // value used by the controller.
        // --------------------------------------------------------------
        if (!vpf_risk_.empty())
        {
            const int h = output.rows;
            const int w = output.cols;

            const double cell_w =
                static_cast<double>(w) /
                static_cast<double>(vpf_grid_cols_);

            const double cell_h =
                static_cast<double>(h) /
                static_cast<double>(vpf_grid_rows_);

            // vpf_risk_ is already intended to lie in [0, 1].
            // Convert it to an 8-bit grid without frame-wise normalisation.
            cv::Mat risk_clamped;
            cv::max(vpf_risk_, 0.0, risk_clamped);
            cv::min(risk_clamped, 1.0, risk_clamped);

            cv::Mat risk_u8;
            risk_clamped.convertTo(risk_u8, CV_8U, 255.0);

            // Enlarge each VPF cell using nearest-neighbour interpolation so
            // the original 16 x 8 cell boundaries remain exact.
            cv::Mat risk_large;
            cv::resize(
                risk_u8,
                risk_large,
                cv::Size(w, h),
                0.0,
                0.0,
                cv::INTER_NEAREST);

            cv::Mat heatmap;
            cv::applyColorMap(
                risk_large,
                heatmap,
                cv::COLORMAP_JET);

            // Use the heat map only where the trusted row band is active.
            // The rest of the camera image remains unchanged.
            cv::Mat band_mask =
                cv::Mat::zeros(h, w, CV_8UC1);

            const int trusted_y_min =
                static_cast<int>(
                    h * floor_row_min_frac_);

            const int trusted_y_max =
                static_cast<int>(
                    h * floor_row_max_frac_);

            cv::rectangle(
                band_mask,
                cv::Point(0, trusted_y_min),
                cv::Point(w - 1, trusted_y_max),
                cv::Scalar(255),
                cv::FILLED);

            // Do not colour cells whose risk is exactly zero.
            cv::Mat positive_risk_mask;
            cv::threshold(
                risk_large,
                positive_risk_mask,
                0,
                255,
                cv::THRESH_BINARY);

            cv::bitwise_and(
                band_mask,
                positive_risk_mask,
                band_mask);

            cv::Mat blended;
            cv::addWeighted(
                output,
                0.58,
                heatmap,
                0.42,
                0.0,
                blended);

            blended.copyTo(output, band_mask);

            // Draw all 16 x 8 cell borders.
            for (int cc = 0; cc <= vpf_grid_cols_; ++cc)
            {
                const int x =
                    static_cast<int>(
                        cc * cell_w);

                cv::line(
                    output,
                    cv::Point(x, 0),
                    cv::Point(x, h - 1),
                    cv::Scalar(220, 220, 220),
                    1,
                    cv::LINE_AA);
            }

            for (int rr = 0; rr <= vpf_grid_rows_; ++rr)
            {
                const int y =
                    static_cast<int>(
                        rr * cell_h);

                cv::line(
                    output,
                    cv::Point(0, y),
                    cv::Point(w - 1, y),
                    cv::Scalar(220, 220, 220),
                    1,
                    cv::LINE_AA);
            }

            // Make the left/right boundary used by the controller obvious.
            const int centre_x =
                static_cast<int>(
                    0.5 * w);

            cv::line(
                output,
                cv::Point(centre_x, 0),
                cv::Point(centre_x, h - 1),
                cv::Scalar(255, 255, 255),
                3,
                cv::LINE_AA);

            // Print the numerical risk inside every non-zero cell.
            for (int rr = 0; rr < vpf_grid_rows_; ++rr)
            {
                for (int cc = 0; cc < vpf_grid_cols_; ++cc)
                {
                    const double risk =
                        static_cast<double>(
                            vpf_risk_.at<float>(rr, cc));

                    if (risk <= 0.0)
                    {
                        continue;
                    }

                    const int x1 =
                        static_cast<int>(
                            cc * cell_w);

                    const int y1 =
                        static_cast<int>(
                            rr * cell_h);

                    std::ostringstream risk_text;
                    risk_text
                        << std::fixed
                        << std::setprecision(2)
                        << risk;

                    cv::putText(
                        output,
                        risk_text.str(),
                        cv::Point(x1 + 3, y1 + 16),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.38,
                        cv::Scalar(255, 255, 255),
                        1,
                        cv::LINE_AA);
                }
            }

            cv::putText(
                output,
                "IMAGE LEFT",
                cv::Point(8, h - 12),
                cv::FONT_HERSHEY_SIMPLEX,
                0.5,
                cv::Scalar(255, 255, 255),
                1,
                cv::LINE_AA);

            cv::putText(
                output,
                "IMAGE RIGHT",
                cv::Point(centre_x + 8, h - 12),
                cv::FONT_HERSHEY_SIMPLEX,
                0.5,
                cv::Scalar(255, 255, 255),
                1,
                cv::LINE_AA);
        }

        // --------------------------------------------------------------
        // Column-gap diagnostic overlay.
        // --------------------------------------------------------------
        if (enable_gap_guidance_ &&
            static_cast<int>(column_risk_.size()) == vpf_grid_cols_)
        {
            const double cell_w =
                static_cast<double>(output.cols) /
                static_cast<double>(std::max(1, vpf_grid_cols_));

            for (int cc = 0; cc < vpf_grid_cols_; ++cc)
            {
                const int x1 =
                    static_cast<int>(cc * cell_w);
                const int x2 =
                    static_cast<int>((cc + 1) * cell_w) - 1;

                std::ostringstream col_text;
                col_text
                    << std::fixed
                    << std::setprecision(2)
                    << column_risk_[cc];

                cv::putText(
                    output,
                    col_text.str(),
                    cv::Point(x1 + 2, output.rows - 32),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.30,
                    cv::Scalar(255, 255, 255),
                    1,
                    cv::LINE_AA);

                (void)x2;
            }

            if (safe_gap_available_ &&
                selected_gap_start_col_ >= 0 &&
                selected_gap_end_col_ >= selected_gap_start_col_)
            {
                const int gap_x1 =
                    static_cast<int>(
                        selected_gap_start_col_ * cell_w);
                const int gap_x2 =
                    static_cast<int>(
                        (selected_gap_end_col_ + 1) * cell_w) - 1;

                cv::rectangle(
                    output,
                    cv::Point(gap_x1, 2),
                    cv::Point(gap_x2, output.rows - 2),
                    cv::Scalar(0, 255, 255),
                    3,
                    cv::LINE_AA);

                const int gap_center_x =
                    static_cast<int>(
                        (selected_gap_center_col_ + 0.5) * cell_w);

                cv::line(
                    output,
                    cv::Point(gap_center_x, 0),
                    cv::Point(gap_center_x, output.rows - 1),
                    cv::Scalar(0, 255, 255),
                    3,
                    cv::LINE_AA);
            }

            if (gap_escape_required_)
            {
                cv::putText(
                    output,
                    "NO SAFE GAP -> ROTATIONAL APF",
                    cv::Point(20, output.rows - 60),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.65,
                    cv::Scalar(0, 0, 255),
                    2,
                    cv::LINE_AA);
            }
        }

        // Dynamic translational FOE and moving frontal-TTC corridor.
        if (dynamic_foe_valid_)
        {
            const int corridor_min = static_cast<int>(
                clamp(frontal_corridor_x_min_px_, 0.0,
                      static_cast<double>(std::max(0, output.cols - 1))));
            const int corridor_max = static_cast<int>(
                clamp(frontal_corridor_x_max_px_, 0.0,
                      static_cast<double>(std::max(0, output.cols - 1))));

            cv::line(output,
                     cv::Point(corridor_min, 0),
                     cv::Point(corridor_min, output.rows - 1),
                     cv::Scalar(255, 0, 255), 1);
            cv::line(output,
                     cv::Point(corridor_max, 0),
                     cv::Point(corridor_max, output.rows - 1),
                     cv::Scalar(255, 0, 255), 1);

            if (dynamic_foe_x_px_ >= 0.0 &&
                dynamic_foe_x_px_ < output.cols &&
                dynamic_foe_y_px_ >= 0.0 &&
                dynamic_foe_y_px_ < output.rows)
            {
                const cv::Point foe_point(
                    static_cast<int>(dynamic_foe_x_px_),
                    static_cast<int>(dynamic_foe_y_px_));
                cv::drawMarker(output, foe_point, cv::Scalar(255, 0, 255),
                               cv::MARKER_CROSS, 18, 2, cv::LINE_AA);
            }
        }

        // Mark the trusted row band so it's visible during tuning.
        cv::line(output, cv::Point(0, static_cast<int>(output.rows * floor_row_min_frac_)),
                 cv::Point(output.cols, static_cast<int>(output.rows * floor_row_min_frac_)),
                 cv::Scalar(255, 255, 0), 1);
        cv::line(output, cv::Point(0, static_cast<int>(output.rows * floor_row_max_frac_)),
                 cv::Point(output.cols, static_cast<int>(output.rows * floor_row_max_frac_)),
                 cv::Scalar(255, 255, 0), 1);

        std::vector<std::string> lines;
        lines.push_back(have_ttc_filtered_ ? formatString("TTC: %.2f s", ttc_filtered_) : "TTC: not reliable");
        lines.push_back(formatString("Features: %d", tracked_features_));
        lines.push_back(formatString("Left flow: %.2f", left_danger_));
        lines.push_back(formatString("Right flow: %.2f", right_danger_));
        lines.push_back(formatString("Rep y (raw): %.2f", repulsive_y_));
        lines.push_back(formatString("Rep y (filt): %.2f", repulsive_y_filtered_));
        lines.push_back(
            std::string("Gap guidance: ") +
            (gap_guidance_active_ ? "ACTIVE" :
             (gap_escape_required_ ? "NO GAP" : "standby")));
        lines.push_back(formatString("Max col risk: %.2f", max_column_risk_));
        lines.push_back(formatString("Gap centre col: %.2f", selected_gap_center_col_));
        lines.push_back(formatString("Gap offset: %.1f deg", selected_gap_heading_offset_rad_ * 180.0 / M_PI));
        lines.push_back(
            std::string("Rot APF: ") +
            (rotational_apf_active_ ? "ACTIVE" :
             (enable_rotational_apf_ ? "standby" : "disabled")));
        lines.push_back(
            std::string("Rot dir: ") +
            ((rotational_direction_ > 0) ? "CW" :
             ((rotational_direction_ < 0) ? "CCW" : "none")));
        lines.push_back(formatString("Rot vx body: %.2f", last_rotational_vx_body_));
        lines.push_back(formatString("Rot vy body: %.2f", last_rotational_vy_body_));
        lines.push_back(formatString("Rot risk: %.2f", last_rotational_risk_strength_));
        lines.push_back(formatString("vx: %.2f", last_vx_));
        lines.push_back(formatString("vy: %.2f", last_vy_));
        lines.push_back(formatString("vz: %.2f", last_vz_));
        lines.push_back(formatString("Yaw: %.2f rad", current_yaw_));
        lines.push_back(formatString("Yaw rate: %.3f rad/s", current_yaw_rate_));
        lines.push_back(formatString("Desired yaw: %.2f rad", desired_heading_));
        lines.push_back(formatString("Heading error: %.1f deg", last_heading_error_ * 180.0 / M_PI));
        lines.push_back(formatString("Body vx: %.2f m/s", current_body_vx_));
        lines.push_back(formatString("Body vy: %.2f m/s", current_body_vy_));
        lines.push_back(
            std::string("Dynamic FOE: ") +
            (dynamic_foe_valid_ ? "VALID" : "UNAVAILABLE"));
        lines.push_back(formatString("FOE x: %.1f px", dynamic_foe_x_px_));
        lines.push_back(formatString("FOE y: %.1f px", dynamic_foe_y_px_));
        lines.push_back(have_frontal_ttc_filtered_ ?
                        formatString("Front TTC: %.2f s", frontal_ttc_filtered_) :
                        "Front TTC: not reliable");
        lines.push_back(
            std::string("Yaw flow comp: ") +
            boolText(enable_yaw_flow_compensation_));
        lines.push_back(std::string("Reached: ") + boolText(reached_goal_));
        lines.push_back(formatString("Osc count: %d", oscillation_count_));
        lines.push_back(formatString("Total osc: %d", total_oscillation_count_));
        lines.push_back(std::string("Osc failure: ") + boolText(oscillation_failure_));
        lines.push_back(std::string("Progress poor: ") + boolText(progress_poor_));
        lines.push_back(formatString("Progress rate: %.2f", progress_rate_));

        if (have_pose_)
        {
            lines.push_back(formatString("x: %.2f", current_pose_.position.x));
            lines.push_back(formatString("y: %.2f", current_pose_.position.y));
            lines.push_back(formatString("z: %.2f", current_pose_.position.z));
        }

        const int y0 = 35;
        for (size_t i = 0; i < lines.size(); ++i)
        {
            cv::putText(output, lines[i], cv::Point(20, y0 + static_cast<int>(35 * i)),
                       cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
        }

        return output;
    }

    void publishVelocity(double vx_world, double vy_world, double vz_world, double yaw_rate = 0.0)
    {
        geometry_msgs::TwistStamped cmd;
        cmd.header.stamp = ros::Time::now();
        cmd.header.frame_id = "world";
        cmd.twist.linear.x = vx_world;
        cmd.twist.linear.y = vy_world;
        cmd.twist.linear.z = vz_world;
        cmd.twist.angular.x = 0.0;
        cmd.twist.angular.y = 0.0;
        cmd.twist.angular.z = yaw_rate;
        vel_pub_.publish(cmd);
    }
    void computeLookaheadPathTarget(
        double current_x,
        double current_y,
        double& target_x,
        double& target_y) const
    {
        const double path_dx = goal_x_ - start_x_;
        const double path_dy = goal_y_ - start_y_;
        const double path_length_sq =
            path_dx * path_dx + path_dy * path_dy;

        if (path_length_sq < 1e-9)
        {
            target_x = goal_x_;
            target_y = goal_y_;
            return;
        }

        const double path_length = std::sqrt(path_length_sq);

        // Project the current UAV position onto the original start-to-goal
        // reference. The scalar projection is expressed as distance along
        // the path, rather than using x alone, so this also works for a
        // diagonal or nearly vertical reference.
        const double from_start_x = current_x - start_x_;
        const double from_start_y = current_y - start_y_;

        double projection_distance =
            (from_start_x * path_dx + from_start_y * path_dy) /
            path_length;

        projection_distance =
            clamp(projection_distance, 0.0, path_length);

        // Move forward from the projection and use that point as the local
        // recovery target. The target can never pass beyond the final goal.
        const double target_distance =
            std::min(
                projection_distance +
                    std::max(0.0, path_lookahead_distance_),
                path_length);

        const double path_unit_x = path_dx / path_length;
        const double path_unit_y = path_dy / path_length;

        target_x = start_x_ + target_distance * path_unit_x;
        target_y = start_y_ + target_distance * path_unit_y;
    }

    void resetPerformanceMetrics()
    {
        first_image_time_ = -1.0;
        last_image_time_ = -1.0;
        previous_image_time_for_dt_ = -1.0;
        image_count_ = 0;
        min_image_dt_ = std::numeric_limits<double>::max();
        max_image_dt_ = 0.0;
        processing_time_sum_ms_ = 0.0;
        max_processing_time_ms_ = 0.0;
        processing_count_ = 0;
    }

    // Estimate one dominant visual obstacle direction from the active VPF.
    //
    // The camera optical axis is body +x and positive image x points to the
    // vehicle's right. ROS body +y points left, hence the minus sign in
    // obstacle_y_body. The VPF risk already contains TTC/looming information,
    // so it is used as the bearing weight.
    bool estimateDominantVisualObstacleBody(
        double& obstacle_x_body,
        double& obstacle_y_body,
        double& risk_strength) const
    {
        obstacle_x_body = 1.0;
        obstacle_y_body = 0.0;
        risk_strength = 0.0;

        if (vpf_risk_.empty() ||
            last_image_width_ <= 0 ||
            vpf_grid_cols_ <= 0)
        {
            return false;
        }

        double weighted_u = 0.0;
        double risk_sum = 0.0;
        double max_risk = 0.0;
        int active_cells = 0;

        for (int row = 0; row < vpf_risk_.rows; ++row)
        {
            for (int col = 0; col < vpf_risk_.cols; ++col)
            {
                const double risk =
                    static_cast<double>(
                        vpf_risk_.at<float>(row, col));

                if (risk <= 0.0)
                {
                    continue;
                }

                const double u =
                    (static_cast<double>(col) + 0.5) /
                    static_cast<double>(vpf_grid_cols_) *
                    static_cast<double>(last_image_width_);

                weighted_u += risk * u;
                risk_sum += risk;
                max_risk = std::max(max_risk, risk);
                active_cells++;
            }
        }

        if (active_cells == 0 || risk_sum <= 1e-6)
        {
            return false;
        }

        const double centroid_u =
            weighted_u / risk_sum;

        const double cx =
            0.5 * static_cast<double>(last_image_width_);

        const double fx =
            (camera_fx_pixels_ > 1.0)
                ? camera_fx_pixels_
                : cx;

        const double bearing_body =
            std::atan2(centroid_u - cx, fx);

        obstacle_x_body =
            std::cos(bearing_body);

        // Positive image bearing means obstacle to body-right, while body +y
        // points left.
        obstacle_y_body =
            -std::sin(bearing_body);

        const double norm =
            std::hypot(obstacle_x_body, obstacle_y_body);

        if (norm <= 1e-6)
        {
            return false;
        }

        obstacle_x_body /= norm;
        obstacle_y_body /= norm;

        const double mean_active_risk =
            risk_sum / static_cast<double>(active_cells);

        // Max risk gives timely response; mean risk prevents one isolated cell
        // from fully dominating the force magnitude.
        risk_strength =
            clamp(
                0.65 * max_risk +
                0.35 * mean_active_risk,
                0.0,
                1.0);

        return true;
    }

    // Compute bounded longitudinal repulsion from the frontal VPF risk.
    //
    // The VPF risk already contains local TTC/looming information. The horizontal
    // obstacle bearing supplies a frontal-alignment factor. The resulting risk is
    // low-pass filtered and converted into a cancellation fraction. Repulsion is
    // saturated at the attractive forward target, so it can stop but not reverse
    // the UAV.
    void computeLongitudinalRepulsion(
        double forward_attraction_target,
        bool& visual_obstacle_valid,
        double& obstacle_x_body,
        double& obstacle_y_body,
        double& visual_risk_strength,
        double& frontal_risk,
        double& longitudinal_repulsive_vx,
        double& forward_resultant_target)
    {
        obstacle_x_body = 1.0;
        obstacle_y_body = 0.0;
        visual_risk_strength = 0.0;

        visual_obstacle_valid =
            estimateDominantVisualObstacleBody(
                obstacle_x_body,
                obstacle_y_body,
                visual_risk_strength);

        double raw_frontal_risk = 0.0;

        if (visual_obstacle_valid)
        {
            last_visual_obstacle_x_body_ = obstacle_x_body;
            last_visual_obstacle_y_body_ = obstacle_y_body;
            last_rotational_risk_strength_ = visual_risk_strength;

            const double frontal_alignment =
                clamp(obstacle_x_body, 0.0, 1.0);

            raw_frontal_risk =
                visual_risk_strength *
                std::pow(
                    frontal_alignment,
                    std::max(0.1, longitudinal_frontal_exponent_));
        }

        // Independent TTC safety evidence for a centred/head-on obstacle.
        // At TTC >= frontal_ttc_danger_ it contributes zero risk; as TTC
        // approaches zero the risk approaches one.
        double frontal_ttc_risk = 0.0;
        if (have_frontal_ttc_filtered_)
        {
            frontal_ttc_risk =
                clamp(
                    1.0 -
                        frontal_ttc_filtered_ /
                        std::max(0.1, frontal_ttc_danger_),
                    0.0,
                    1.0);

            raw_frontal_risk =
                std::max(raw_frontal_risk, frontal_ttc_risk);

            // If the thresholded VPF contains no positive cells but the central
            // TTC channel sees a close looming object, represent it as a centred
            // frontal obstacle.  This lets longitudinal braking and rotational
            // APF react instead of requiring the VPF centroid to exist first.
            if (!visual_obstacle_valid && frontal_ttc_risk > 0.0)
            {
                visual_obstacle_valid = true;
                obstacle_x_body = 1.0;
                obstacle_y_body = 0.0;
                visual_risk_strength = frontal_ttc_risk;

                last_visual_obstacle_x_body_ = obstacle_x_body;
                last_visual_obstacle_y_body_ = obstacle_y_body;
                last_rotational_risk_strength_ = visual_risk_strength;
            }
            else if (visual_obstacle_valid)
            {
                visual_risk_strength =
                    std::max(visual_risk_strength, frontal_ttc_risk);
                last_rotational_risk_strength_ = visual_risk_strength;
            }
        }

        if (enable_longitudinal_repulsion_)
        {
            longitudinal_frontal_risk_filtered_ =
                longitudinal_repulsion_filter_alpha_ *
                    raw_frontal_risk +
                (1.0 - longitudinal_repulsion_filter_alpha_) *
                    longitudinal_frontal_risk_filtered_;
        }
        else
        {
            longitudinal_frontal_risk_filtered_ = 0.0;
        }

        frontal_risk =
            clamp(longitudinal_frontal_risk_filtered_, 0.0, 1.0);

        const double visual_cancellation_fraction =
            enable_longitudinal_repulsion_
                ? clamp(
                      k_longitudinal_repulsion_ * frontal_risk,
                      0.0,
                      1.0)
                : 0.0;

        // The dedicated frontal TTC path is a safety override.  It remains
        // continuous (no hard stop/go thresholds) but is not weakened by the
        // ordinary VPF gain when a head-on collision becomes imminent.
        const double cancellation_fraction =
            enable_longitudinal_repulsion_
                ? std::max(
                      visual_cancellation_fraction,
                      frontal_ttc_risk)
                : 0.0;

        longitudinal_repulsive_vx =
            -forward_attraction_target * cancellation_fraction;

        forward_resultant_target =
            std::max(
                0.0,
                forward_attraction_target +
                    longitudinal_repulsive_vx);

        // When the dedicated frontal TTC channel reports an imminent
        // obstacle, reduce forward speed quadratically with TTC.
        // This is continuous: at the danger horizon it changes nothing,
        // while very small TTC drives the allowed forward target near zero.
        if (enable_longitudinal_repulsion_ &&
            have_frontal_ttc_filtered_ &&
            frontal_ttc_filtered_ < frontal_ttc_danger_)
        {
            const double ttc_scale =
                clamp(
                    frontal_ttc_filtered_ /
                        std::max(0.1, frontal_ttc_danger_),
                    0.0,
                    1.0);

            const double frontal_speed_limit =
                forward_attraction_target *
                std::pow(ttc_scale, 1.5);

            forward_resultant_target =
                std::min(
                    forward_resultant_target,
                    frontal_speed_limit);

            // Keep diagnostics physically consistent after the TTC cap.
            longitudinal_repulsive_vx =
                forward_resultant_target -
                forward_attraction_target;
        }

        last_forward_attraction_target_ =
            forward_attraction_target;
        last_longitudinal_frontal_risk_ =
            frontal_risk;
        last_longitudinal_repulsive_vx_ =
            longitudinal_repulsive_vx;
        last_forward_resultant_target_ =
            forward_resultant_target;
    }

    // Activate/deactivate rotational APF from a persistent longitudinal local
    // minimum. Low resultant forward speed is required only to START an episode;
    // once active, visual obstacle influence maintains the locked tangential
    // direction until the release delay expires.
    void updateRotationalAPFState(
        double x,
        double y,
        bool danger_active,
        bool visual_obstacle_valid,
        double obstacle_x_body,
        double obstacle_y_body,
        double risk_strength,
        double frontal_risk,
        double forward_resultant_target,
        double now)
    {
        if (!enable_rotational_apf_)
        {
            rotational_blockage_active_ = false;
            rotational_blockage_start_time_ = 0.0;
            last_rotational_blockage_duration_ = 0.0;

            if (rotational_apf_active_)
            {
                rotational_apf_active_ = false;
                rotational_direction_ = 0;
            }
            return;
        }

        if (visual_obstacle_valid)
        {
            last_visual_obstacle_x_body_ = obstacle_x_body;
            last_visual_obstacle_y_body_ = obstacle_y_body;
            last_rotational_risk_strength_ = risk_strength;
        }

        const bool trap_trigger_satisfied =
            !rotational_trigger_on_trap_ ||
            oscillation_active_ ||
            progress_poor_;

        const bool obstacle_danger_confirmed =
            danger_active ||
            frontal_risk >= rotational_activation_risk_;

        const bool longitudinal_local_minimum =
            !safe_gap_available_ &&
            obstacle_danger_confirmed &&
            visual_obstacle_valid &&
            risk_strength >= rotational_activation_risk_ &&
            frontal_risk >= rotational_activation_risk_ &&
            forward_resultant_target <= rotational_start_forward_speed_ &&
            trap_trigger_satisfied;

        // Dense-clutter escape trigger: when the column-wise VPF contains no
        // sufficiently wide safe corridor, do not wait for longitudinal speed
        // to collapse into a classical APF local minimum. Request the tangential
        // rotational field directly, while retaining the existing confirmation
        // delay below to reject one-frame visual noise.
        const bool recent_no_safe_gap =
            have_last_no_safe_gap_time_ &&
            (now - last_no_safe_gap_time_) <=
                gap_no_safe_latch_s_;

        const bool no_safe_gap_escape =
            enable_gap_guidance_ &&
            (gap_escape_required_ || recent_no_safe_gap) &&
            max_column_risk_ >= gap_rotational_trigger_risk_threshold_ &&
            visual_obstacle_valid &&
            obstacle_danger_confirmed;

        const bool rotational_start_condition =
            longitudinal_local_minimum ||
            no_safe_gap_escape;

        if (!rotational_apf_active_)
        {
            if (rotational_start_condition)
            {
                if (!rotational_blockage_active_)
                {
                    rotational_blockage_active_ = true;
                    rotational_blockage_start_time_ = now;
                }

                last_rotational_blockage_duration_ =
                    now - rotational_blockage_start_time_;

                if (last_rotational_blockage_duration_ >=
                    rotational_start_delay_s_)
                {
                    const double phi =
                        std::atan2(
                            goal_y_ - y,
                            goal_x_ - x);

                    const double obstacle_bearing_body =
                        std::atan2(
                            obstacle_y_body,
                            obstacle_x_body);

                    const double rho =
                        normalizeAngle(
                            current_yaw_ +
                            obstacle_bearing_body);

                    const double theta =
                        normalizeAngle(phi - rho);

                    if (no_safe_gap_escape)
                    {
                        // +1 produces a body-left tangent for a centred frontal
                        // obstacle; -1 produces body-right. Use the established
                        // least-dangerous side when the image contains no gap.
                        rotational_direction_ =
                            (last_avoid_direction_ >= 0.0) ? 1 : -1;
                    }
                    else if (std::abs(theta) <=
                             rotational_direction_deadband_rad_)
                    {
                        // For a centred obstacle, retain the established lateral
                        // preference instead of allowing tiny centroid noise to
                        // flip the selected rotation direction.
                        rotational_direction_ =
                            (last_avoid_direction_ >= 0.0) ? 1 : -1;
                    }
                    else
                    {
                        // +1 denotes clockwise R=[0 1; -1 0].
                        rotational_direction_ =
                            (theta >= 0.0) ? 1 : -1;
                    }

                    rotational_apf_active_ = true;
                    rotational_apf_used_ = true;
                    rotational_activation_count_++;
                    rotational_last_request_time_ = now;
                    rotational_blockage_active_ = false;
                    rotational_safe_gap_reacquire_active_ = false;
                    rotational_safe_gap_reacquire_start_time_ = 0.0;

                    // The tangential controller is a new avoidance mode. Do not
                    // let reversals accumulated by the baseline VPF terminate it
                    // immediately; evaluate the new mode from a fresh episode.
                    clearOscillationEpisode();
                    oscillation_failure_ = false;

                    ROS_WARN(
                        "Rotational APF activated: cause=%s direction=%s, theta=%.3f rad, visual_risk=%.3f, frontal_risk=%.3f, resultant_vx=%.3f m/s.",
                        no_safe_gap_escape ? "NO_SAFE_GAP" : "LONGITUDINAL_LOCAL_MINIMUM",
                        rotational_direction_ > 0 ? "CW" : "CCW",
                        theta,
                        risk_strength,
                        frontal_risk,
                        forward_resultant_target);
                }
            }
            else
            {
                rotational_blockage_active_ = false;
                rotational_blockage_start_time_ = 0.0;
                last_rotational_blockage_duration_ = 0.0;
            }

            return;
        }

        // If a valid visual corridor reappears and stays valid, rotational APF
        // should relinquish control. Previously it stayed active merely because
        // *some* obstacle risk remained in view, even while GAP_SELECT was
        // repeatedly finding a usable corridor. That made the vehicle keep
        // curling away from the route through multiple unrelated obstacles.
        const bool stable_gap_candidate =
            enable_gap_guidance_ &&
            gap_guidance_active_ &&
            safe_gap_available_ &&
            !gap_escape_required_;

        if (stable_gap_candidate)
        {
            if (!rotational_safe_gap_reacquire_active_)
            {
                rotational_safe_gap_reacquire_active_ = true;
                rotational_safe_gap_reacquire_start_time_ = now;
            }
            else if ((now - rotational_safe_gap_reacquire_start_time_) >=
                     rotational_safe_gap_release_delay_s_)
            {
                ROS_INFO(
                    "Rotational APF released after stable safe gap was reacquired for %.2f s.",
                    now - rotational_safe_gap_reacquire_start_time_);

                rotational_apf_active_ = false;
                rotational_direction_ = 0;
                rotational_blockage_active_ = false;
                rotational_blockage_start_time_ = 0.0;
                last_rotational_blockage_duration_ = 0.0;
                rotational_safe_gap_reacquire_active_ = false;
                rotational_safe_gap_reacquire_start_time_ = 0.0;

                // Do not carry a stale tangential command into the newly
                // reacquired gap-guidance mode.
                rotational_vx_filtered_ = 0.0;
                rotational_vy_filtered_ = 0.0;
                return;
            }
        }
        else
        {
            rotational_safe_gap_reacquire_active_ = false;
            rotational_safe_gap_reacquire_start_time_ = 0.0;
        }

        const bool maintain_rotational_request =
            visual_obstacle_valid &&
            risk_strength >= rotational_maintain_risk_;

        if (maintain_rotational_request)
        {
            rotational_last_request_time_ = now;
        }

        if (!maintain_rotational_request &&
            (now - rotational_last_request_time_) >=
                rotational_release_delay_s_)
        {
            ROS_INFO(
                "Rotational APF released after visual obstacle influence decreased.");

            rotational_apf_active_ = false;
            rotational_direction_ = 0;
            rotational_blockage_active_ = false;
            rotational_blockage_start_time_ = 0.0;
            last_rotational_blockage_duration_ = 0.0;
            rotational_safe_gap_reacquire_active_ = false;
            rotational_safe_gap_reacquire_start_time_ = 0.0;
        }
    }

    // Compute the +/-90-degree tangential vector in the BODY X-Y plane.
    //
    // radial = UAV - obstacle (points away from obstacle)
    // clockwise tangent        = [ radial_y, -radial_x ]
    // counter-clockwise tangent= [-radial_y,  radial_x ]
    //
    // Both components are returned. The tangent may contain a negative x
    // component, but the final published forward command is clamped at zero so
    // the rotational term cannot make the UAV reverse.
    void computeRotationalForceBody(
        double& rotational_vx_body,
        double& rotational_vy_body)
    {
        double target_vx_body = 0.0;
        double target_vy_body = 0.0;

        if (enable_rotational_apf_ &&
            rotational_apf_active_ &&
            rotational_direction_ != 0)
        {
            double obstacle_x_body =
                last_visual_obstacle_x_body_;

            double obstacle_y_body =
                last_visual_obstacle_y_body_;

            double current_risk =
                last_rotational_risk_strength_;

            double measured_x = 1.0;
            double measured_y = 0.0;
            double measured_risk = 0.0;

            if (estimateDominantVisualObstacleBody(
                    measured_x,
                    measured_y,
                    measured_risk))
            {
                obstacle_x_body = measured_x;
                obstacle_y_body = measured_y;
                current_risk = measured_risk;

                last_visual_obstacle_x_body_ =
                    measured_x;

                last_visual_obstacle_y_body_ =
                    measured_y;

                last_rotational_risk_strength_ =
                    measured_risk;
            }

            const double radial_x_body =
                -obstacle_x_body;

            const double radial_y_body =
                -obstacle_y_body;

            double tangent_x_body = 0.0;
            double tangent_y_body = 0.0;

            if (rotational_direction_ > 0)
            {
                // Clockwise rotation.
                tangent_x_body = radial_y_body;
                tangent_y_body = -radial_x_body;
            }
            else
            {
                // Counter-clockwise rotation.
                tangent_x_body = -radial_y_body;
                tangent_y_body = radial_x_body;
            }

            const double tangent_norm =
                std::hypot(
                    tangent_x_body,
                    tangent_y_body);

            const double radial_norm =
                std::hypot(
                    radial_x_body,
                    radial_y_body);

            if (tangent_norm > 1e-6 && radial_norm > 1e-6)
            {
                // Unit tangent: moves around the obstacle.
                tangent_x_body /= tangent_norm;
                tangent_y_body /= tangent_norm;

                // Unit radial: points directly away from the obstacle.
                const double outward_x_body =
                    radial_x_body / radial_norm;

                const double outward_y_body =
                    radial_y_body / radial_norm;

                // A pure tangent preserves radius. Add a controlled outward
                // component so rotational escape also grows obstacle clearance.
                double escape_x_body =
                    tangent_x_body +
                    rotational_radial_escape_weight_ * outward_x_body;

                double escape_y_body =
                    tangent_y_body +
                    rotational_radial_escape_weight_ * outward_y_body;

                const double escape_norm =
                    std::hypot(
                        escape_x_body,
                        escape_y_body);

                if (escape_norm > 1e-6)
                {
                    escape_x_body /= escape_norm;
                    escape_y_body /= escape_norm;

                    const double rotational_speed =
                        clamp(
                            k_rotational_ * current_risk,
                            rotational_min_speed_,
                            rotational_max_speed_);

                    target_vx_body =
                        rotational_speed *
                        escape_x_body;

                    target_vy_body =
                        rotational_speed *
                        escape_y_body;
                }
            }
        }

        rotational_vx_filtered_ =
            rotational_filter_alpha_ *
                target_vx_body +
            (1.0 - rotational_filter_alpha_) *
                rotational_vx_filtered_;

        rotational_vy_filtered_ =
            rotational_filter_alpha_ *
                target_vy_body +
            (1.0 - rotational_filter_alpha_) *
                rotational_vy_filtered_;

        if (!rotational_apf_active_)
        {
            if (std::abs(rotational_vx_filtered_) < 1e-3)
            {
                rotational_vx_filtered_ = 0.0;
            }

            if (std::abs(rotational_vy_filtered_) < 1e-3)
            {
                rotational_vy_filtered_ = 0.0;
            }
        }

        rotational_vx_body =
            rotational_vx_filtered_;

        rotational_vy_body =
            rotational_vy_filtered_;

        last_rotational_vx_body_ =
            rotational_vx_body;

        last_rotational_vy_body_ =
            rotational_vy_body;
    }

    void applyVelocityHeadingControl(
        double desired_vx_world,
        double desired_vy_world,
        double& commanded_vx_world,
        double& commanded_vy_world,
        double& yaw_rate_command)
    {
        commanded_vx_world = desired_vx_world;
        commanded_vy_world = desired_vy_world;
        yaw_rate_command =
            clamp(
                -k_yaw_recover_ * current_yaw_,
                -max_yaw_recover_rate_,
                max_yaw_recover_rate_);

        const double horizontal_speed =
            std::hypot(
                desired_vx_world,
                desired_vy_world);

        if (!enable_velocity_heading_control_ ||
            horizontal_speed <= heading_min_horizontal_speed_)
        {
            desired_heading_ = current_yaw_;
            last_heading_error_ = 0.0;
            last_motion_alignment_scale_ = 1.0;
            return;
        }

        desired_heading_ =
            std::atan2(
                desired_vy_world,
                desired_vx_world);

        last_heading_error_ =
            normalizeAngle(
                desired_heading_ - current_yaw_);

        yaw_rate_command =
            clamp(
                k_velocity_heading_ * last_heading_error_,
                -max_velocity_heading_rate_,
                max_velocity_heading_rate_);

        const double abs_heading_error =
            std::abs(last_heading_error_);

        const double slow_angle =
            heading_slow_angle_deg_ * M_PI / 180.0;

        const double strong_slow_angle =
            heading_strong_slow_angle_deg_ * M_PI / 180.0;

        double motion_scale = 1.0;
        if (abs_heading_error >= strong_slow_angle)
        {
            motion_scale =
                clamp(
                    heading_strong_slow_scale_,
                    0.0,
                    1.0);
        }
        else if (abs_heading_error >= slow_angle)
        {
            motion_scale =
                clamp(
                    heading_slow_scale_,
                    0.0,
                    1.0);
        }

        last_motion_alignment_scale_ = motion_scale;

        // World -> body. The camera is fixed along body +x.
        double desired_vx_body =
            std::cos(current_yaw_) * desired_vx_world +
            std::sin(current_yaw_) * desired_vy_world;

        double desired_vy_body =
            -std::sin(current_yaw_) * desired_vx_world +
             std::cos(current_yaw_) * desired_vy_world;

        desired_vx_body *= motion_scale;
        desired_vy_body *= motion_scale;

        // Never command reverse flight relative to the forward camera. Limit
        // sideways crabbing until yaw has aligned the camera with the motion.
        desired_vx_body =
            clamp(
                desired_vx_body,
                0.0,
                max_forward_speed_);

        desired_vy_body =
            clamp(
                desired_vy_body,
                -max_body_lateral_speed_,
                max_body_lateral_speed_);

        // Body -> world for the existing world-frame velocity interface.
        commanded_vx_world =
            std::cos(current_yaw_) * desired_vx_body -
            std::sin(current_yaw_) * desired_vy_body;

        commanded_vy_world =
            std::sin(current_yaw_) * desired_vx_body +
            std::cos(current_yaw_) * desired_vy_body;

        // Body-forward motion has already been clamped non-negative above.
        // In world coordinates, however, a forward-facing vehicle may need a
        // negative x velocity after turning around or recovering from overshoot.
        commanded_vx_world =
            clamp(
                commanded_vx_world,
                -max_forward_speed_,
                max_forward_speed_);

        commanded_vy_world =
            clamp(
                commanded_vy_world,
                -max_lateral_speed_,
                max_lateral_speed_);
    }

    void controlLoop(const ros::TimerEvent&)
    {
        if (experiment_complete_ || resetting_world_ || attempt_finished_ || !have_pose_) { return; }

        if (stabilizing_after_reset_)
        {
            publishVelocity(0.0, 0.0, 0.0);
            if (ros::Time::now().toSec() < stabilization_end_time_) { return; }

            stabilizing_after_reset_ = false;
            have_prev_image_ = false;
            prev_gray_.release();
            prev_pts_.clear();
            prev_time_ = 0.0;
            have_ttc_filtered_ = false;
            ttc_filtered_ = 0.0;
            ttc_miss_count_ = 0;
            have_frontal_ttc_filtered_ = false;
            frontal_ttc_filtered_ = 0.0;
            frontal_ttc_miss_count_ = 0;
            repulsive_y_ = 0.0;
            repulsive_y_filtered_ = 0.0;
            left_danger_ = 0.0;
            right_danger_ = 0.0;
            tracked_features_ = 0;
            current_yaw_rate_ = 0.0;
            current_body_vx_ = 0.0;
            current_body_vy_ = 0.0;
            current_body_vz_ = 0.0;
            desired_heading_ = current_yaw_;
            last_heading_error_ = 0.0;
            last_motion_alignment_scale_ = 1.0;
            centre_foe_model_valid_ = false;
            last_centre_foe_valid_time_ = 0.0;
            have_last_centre_foe_valid_time_ = false;
            dynamic_foe_valid_ = false;
            dynamic_foe_x_px_ = 320.0;
            dynamic_foe_y_px_ = 240.0;
            frontal_corridor_x_min_px_ = 0.0;
            frontal_corridor_x_max_px_ = 0.0;
            vx_command_ = 0.0;
            vy_filtered_ = 0.0;
            longitudinal_frontal_risk_filtered_ = 0.0;
            last_forward_attraction_target_ = 0.0;
            last_longitudinal_frontal_risk_ = 0.0;
            last_longitudinal_repulsive_vx_ = 0.0;
            last_forward_resultant_target_ = 0.0;
            resetGapGuidanceState();

            rotational_apf_active_ = false;
            rotational_direction_ = 0;
            rotational_last_request_time_ = 0.0;
            rotational_blockage_active_ = false;
            rotational_blockage_start_time_ = 0.0;
            last_rotational_blockage_duration_ = 0.0;
            rotational_vx_filtered_ = 0.0;
            rotational_vy_filtered_ = 0.0;
            last_rotational_vx_body_ = 0.0;
            last_rotational_vy_body_ = 0.0;
            last_rotational_risk_strength_ = 0.0;

            movement_watch_started_ = false;

            ROS_INFO("Post-reset stabilization complete. Starting run %d / %d.", current_run_, target_runs_);
        }

        const double x = current_pose_.position.x;
        const double y = current_pose_.position.y;
        const double z = current_pose_.position.z;

        bool collision_now = false;
        current_obstacle_clearance_ = std::numeric_limits<double>::max();

        for (const auto& obs : obstacles_)
        {
            const double ox = obs.first;
            const double oy = obs.second;
            const double dx = x - ox;
            const double dy = y - oy;
            const double centre_distance = std::sqrt(dx * dx + dy * dy);
            const double clearance = centre_distance - (uav_collision_radius_ + obstacle_radius_);

            current_obstacle_clearance_ = std::min(current_obstacle_clearance_, clearance);

            if (takeoff_done_ && !reached_goal_ && !oscillation_failure_)
            {
                if (clearance < minimum_obstacle_clearance_)
                {
                    minimum_obstacle_clearance_ = clearance;
                    minimum_clearance_obstacle_x_ = ox;
                    minimum_clearance_obstacle_y_ = oy;
                }
                if (clearance > 0.0 && clearance < minimum_collision_free_clearance_)
                {
                    minimum_collision_free_clearance_ = clearance;
                    minimum_safe_clearance_obstacle_x_ = ox;
                    minimum_safe_clearance_obstacle_y_ = oy;
                }
            }
            if (clearance <= 0.0) { collision_now = true; }
        }

        if (collision_now && !in_collision_)
        {
            collision_count_++;
            ROS_WARN("Collision #%d detected!", collision_count_);
        }
        in_collision_ = collision_now;

        // -------------------------
        // Takeoff phase
        // -------------------------
        if (!takeoff_done_)
        {
            vx_command_ = 0.0;
            const double error_z = goal_z_ - z;
            double vz = clamp(k_z_ * error_z, -max_vertical_speed_, max_vertical_speed_);

            last_vx_ = 0.0;
            last_vy_ = 0.0;
            last_vz_ = vz;
            publishVelocity(0.0, 0.0, vz);

            if (std::abs(error_z) < 0.2)
            {
                takeoff_done_ = true;
                mission_start_time_ = ros::Time::now().toSec();
                mission_timer_started_ = true;
                path_length_ = 0.0;
                have_last_path_pose_ = true;
                last_path_x_ = x; last_path_y_ = y; last_path_z_ = z;

                movement_watch_started_ = true;
                last_movement_time_ = ros::Time::now().toSec();
                movement_anchor_x_ = x;
                movement_anchor_y_ = y;

                ROS_INFO("Takeoff altitude reached. Starting forward flight.");
                resetGapGuidanceState();
                resetPerformanceMetrics();
            }
            return;
        }

        const double distance_to_goal =
            std::sqrt((goal_x_ - x) * (goal_x_ - x) + (goal_y_ - y) * (goal_y_ - y));
        const bool global_danger_active =
            have_ttc_filtered_ &&
            (ttc_filtered_ < danger_ttc_);

        const bool frontal_danger_active =
            have_frontal_ttc_filtered_ &&
            (frontal_ttc_filtered_ < frontal_ttc_danger_);

        const bool danger_active =
            global_danger_active ||
            frontal_danger_active;

        ROS_INFO_THROTTLE(
            0.5,
            "DANGER_STATE: global=%d global_ttc=%.3f frontal=%d front_ttc=%.3f combined=%d",
            static_cast<int>(global_danger_active),
            have_ttc_filtered_ ? ttc_filtered_ : -1.0,
            static_cast<int>(frontal_danger_active),
            have_frontal_ttc_filtered_ ? frontal_ttc_filtered_ : -1.0,
            static_cast<int>(danger_active));

        const double now = ros::Time::now().toSec();
        if (!movement_watch_started_)
        {
            movement_watch_started_ = true;
            last_movement_time_ = now;
            movement_anchor_x_ = x;
            movement_anchor_y_ = y;
        }

        const double movement_from_anchor = std::hypot(x - movement_anchor_x_, y - movement_anchor_y_);
        if (movement_from_anchor >= movement_threshold_m_)
        {
            movement_anchor_x_ = x;
            movement_anchor_y_ = y;
            last_movement_time_ = now;
        }
        else if ((now - last_movement_time_) >= no_movement_timeout_s_)
        {
            ROS_WARN("Run %d stopped: no meaningful movement for %.1f s.", current_run_, no_movement_timeout_s_);
            finishMission("No movement timeout", false);
            return;
        }

        updateProgressDetector(distance_to_goal, danger_active);

        // Goal completion must be based on actual Euclidean position. The old
        // x-passed + y-close shortcut could declare success several metres past
        // the goal after an avoidance excursion.
        if (distance_to_goal < goal_tolerance_)
        {
            reached_goal_ = true;
        }

        if (reached_goal_)
        {
            const bool collision_free_success = (collision_count_ == 0);
            finishMission(collision_free_success ? "Goal reached" : "Goal reached with collision",
                         collision_free_success);
            return;
        }

        if (oscillation_failure_)
        {
            finishMission("Persistent oscillation", false);
            return;
        }

        // ------------------------------------------------------
        // Forward attraction + bounded longitudinal repulsion
        // ------------------------------------------------------
        double forward_attraction_target = forward_speed_;
        if (distance_to_goal < 8.0)
        {
            forward_attraction_target =
                std::min(forward_attraction_target, 5.0);
        }
        if (distance_to_goal < 4.0)
        {
            forward_attraction_target =
                std::min(forward_attraction_target, 3.0);
        }
        if (distance_to_goal < 2.0)
        {
            forward_attraction_target =
                std::min(forward_attraction_target, 1.0);
        }
        if (distance_to_goal < 0.5)
        {
            forward_attraction_target = 0.3;
        }
        // Do not permanently zero forward attraction merely because world x
        // has passed goal_x_. After a large avoidance manoeuvre, the UAV may be
        // past the goal plane while still far away in y. Goal-return handling
        // below will command a bounded vector back toward the true goal.

        forward_attraction_target =
            clamp(
                forward_attraction_target,
                0.0,
                max_forward_speed_);

        bool visual_obstacle_valid = false;
        double obstacle_x_body = 1.0;
        double obstacle_y_body = 0.0;
        double visual_risk_strength = 0.0;
        double frontal_risk = 0.0;
        double longitudinal_repulsive_vx = 0.0;
        double forward_resultant_target =
            forward_attraction_target;

        computeLongitudinalRepulsion(
            forward_attraction_target,
            visual_obstacle_valid,
            obstacle_x_body,
            obstacle_y_body,
            visual_risk_strength,
            frontal_risk,
            longitudinal_repulsive_vx,
            forward_resultant_target);

        updateRotationalAPFState(
            x,
            y,
            danger_active,
            visual_obstacle_valid,
            obstacle_x_body,
            obstacle_y_body,
            visual_risk_strength,
            frontal_risk,
            forward_resultant_target,
            now);

        // Rotational escape must own the motion envelope. Without this cap, a
        // momentary TTC recovery could let the 10 m/s attraction accelerate
        // back to several m/s while the rotational correction was still only
        // ~1 m/s, which overwhelmed the escape vector and caused collisions.
        if (rotational_apf_active_)
        {
            forward_resultant_target =
                std::min(
                    forward_resultant_target,
                    rotational_forward_speed_cap_);
        }

        // Acceleration and braking are limited separately. The larger
        // deceleration limit allows genuine frontal risk to reduce vx promptly.
        const double dt = 0.01;

        if (forward_resultant_target > vx_command_)
        {
            const double max_accel_delta =
                max_forward_acceleration_ * dt;

            vx_command_ =
                std::min(
                    vx_command_ + max_accel_delta,
                    forward_resultant_target);
        }
        else
        {
            const double max_decel_delta =
                max_forward_deceleration_ * dt;

            vx_command_ =
                std::max(
                    vx_command_ - max_decel_delta,
                    forward_resultant_target);
        }

        double vx =
            clamp(vx_command_, 0.0, max_forward_speed_);

        ROS_INFO_THROTTLE(
            0.5,
            "LONG_REP: valid=%d visual_risk=%.3f frontal_risk=%.3f attraction_vx=%.3f repulsive_vx=%.3f resultant_target=%.3f command_vx=%.3f blockage_age=%.2f rot_active=%d",
            static_cast<int>(visual_obstacle_valid),
            visual_risk_strength,
            frontal_risk,
            forward_attraction_target,
            longitudinal_repulsive_vx,
            forward_resultant_target,
            vx,
            last_rotational_blockage_duration_,
            static_cast<int>(rotational_apf_active_));

        // ------------------------------------------------------
        // APF attraction term (world-frame, path-following)
        // ------------------------------------------------------
        double v_att_y = 0.0;
        if (use_path_controller_)
        {
            double local_target_x = goal_x_;
            double local_target_y = goal_y_;

            computeLookaheadPathTarget(
                x,
                y,
                local_target_x,
                local_target_y);

            // last_y_ref_ remains in the existing CSV output, but now records
            // the y coordinate of the moving look-ahead target rather than the
            // y coordinate of the closest point on the original line.
            last_y_ref_ = local_target_y;
            last_cross_track_error_ = local_target_y - y;

            // Guide gradually toward the look-ahead target. Limiting this term
            // prevents route recovery from overpowering visual/APF avoidance.
            last_v_path_y_ =
                clamp(
                    k_path_y_ * (local_target_y - y),
                    -max_path_recovery_speed_,
                    max_path_recovery_speed_);

            // Keep only a small final-goal bias. It is scaled together with the
            // look-ahead attraction while danger or rotational escape is active.
            last_v_goal_y_ =
                k_goal_y_ * (goal_y_ - y);

            // Keep a SMALL attractive component during danger instead of
            // switching attraction completely off. In safe-gap mode the direct
            // world-frame attraction below is not used for steering; goal bias is
            // incorporated explicitly inside the selected gap. For no-gap and
            // rotational escape modes a weak conventional attraction remains.
            double attraction_scale = 1.0;

            if (gap_guidance_active_ && safe_gap_available_ &&
                !rotational_apf_active_)
            {
                attraction_scale = 0.15;
            }
            else if (rotational_apf_active_)
            {
                attraction_scale =
                    clamp(rotational_attraction_scale_, 0.0, 1.0);
            }
            else if (gap_escape_required_ || danger_active)
            {
                attraction_scale =
                    clamp(avoidance_attraction_scale_, 0.0, 1.0);
            }

            v_att_y =
                attraction_scale *
                (last_v_path_y_ + last_v_goal_y_);

            ROS_INFO_THROTTLE(
                0.5,
                "LOOKAHEAD_PATH: target=(%.2f, %.2f) pos=(%.2f, %.2f) "
                "y_error=%.2f path_vy=%.2f goal_vy=%.2f scale=%.2f",
                local_target_x,
                local_target_y,
                x,
                y,
                last_cross_track_error_,
                last_v_path_y_,
                last_v_goal_y_,
                attraction_scale);
        }
        else
        {
            last_y_ref_ = goal_y_;
            last_cross_track_error_ = goal_y_ - y;
            last_v_path_y_ = 0.0;
            last_v_goal_y_ = k_att_y_ * (goal_y_ - y);
            v_att_y = last_v_goal_y_;
        }

        // ------------------------------------------------------
        // APF repulsion term. The VPF correction is constructed in body
        // coordinates and later transformed into the world command frame.
        // ------------------------------------------------------
        double v_rep_y =
            ((gap_guidance_active_ && safe_gap_available_) ||
             rotational_apf_active_)
                ? 0.0
                : k_rep_y_ * repulsive_y_filtered_;

        // Fallback nudge: danger present but not clearly left/right (e.g.
        // something dead ahead) -> commit to the last meaningful avoidance
        // direction rather than dithering near zero.
        if (!rotational_apf_active_ &&
            !(gap_guidance_active_ && safe_gap_available_) &&
            (danger_active || (left_danger_ + right_danger_) > 0.5) &&
            std::abs(repulsive_y_filtered_) < 0.15)
        {
            v_rep_y +=
                last_avoid_direction_ *
                default_avoid_y_speed_;
        }

        double v_lateral =
            clamp(
                v_rep_y,
                -max_lateral_speed_,
                max_lateral_speed_);

        vy_filtered_ =
            vy_alpha_ * v_lateral +
            (1.0 - vy_alpha_) * vy_filtered_;

        v_lateral =
            vy_filtered_;

        // Once the rotational APF owns the escape manoeuvre, do not stack a
        // decaying legacy left/right command on top of the tangential force.
        if (rotational_apf_active_)
        {
            vy_filtered_ = 0.0;
            v_lateral = 0.0;
        }

        double rotational_vx_body = 0.0;
        double rotational_vy_body = 0.0;

        computeRotationalForceBody(
            rotational_vx_body,
            rotational_vy_body);

        // Combine the ordinary lateral VPF term with the true X-Y tangential
        // rotational term in BODY coordinates.
        const double correction_x_body =
            rotational_vx_body;

        const double correction_y_body =
            clamp(
                v_lateral +
                    rotational_vy_body,
                -max_lateral_speed_,
                max_lateral_speed_);

        // Body -> world:
        // [vx_w] = [ cos(yaw) -sin(yaw)] [vx_b]
        // [vy_w]   [ sin(yaw)  cos(yaw)] [vy_b]
        const double correction_vx_world =
            std::cos(current_yaw_) *
                correction_x_body -
            std::sin(current_yaw_) *
                correction_y_body;

        const double correction_vy_world =
            std::sin(current_yaw_) *
                correction_x_body +
            std::cos(current_yaw_) *
                correction_y_body;

        // During no-gap / rotational avoidance retain goal/path attraction only
        // when it does not directly oppose the avoidance correction. This keeps
        // some progress toward the mission goal without letting the global path
        // pull the UAV back into the obstacle that the local controller is
        // currently trying to bypass.
        if ((gap_escape_required_ || rotational_apf_active_ || danger_active) &&
            std::abs(correction_vy_world) > 0.05 &&
            v_att_y * correction_vy_world < 0.0)
        {
            v_att_y = 0.0;
        }

        double desired_vx_world =
            clamp(
                vx + correction_vx_world,
                0.0,
                max_forward_speed_);

        double desired_vy_world =
            clamp(
                v_att_y + correction_vy_world,
                -max_lateral_speed_,
                max_lateral_speed_);

        // Explicit free-gap steering. During a valid gap episode the selected
        // image corridor defines the desired horizontal motion direction. This
        // makes the heading controller point the camera through the gap instead
        // of deriving only a small yaw angle from left-minus-right repulsion.
        if (enable_gap_guidance_ &&
            gap_guidance_active_ &&
            safe_gap_available_ &&
            !rotational_apf_active_)
        {
            // --------------------------------------------------
            // Goal-biased steering constrained to remain INSIDE the
            // selected free gap.
            // --------------------------------------------------
            const double goal_heading_world =
                std::atan2(
                    goal_y_ - y,
                    goal_x_ - x);

            const double goal_heading_offset_body =
                normalizeAngle(
                    goal_heading_world - current_yaw_);

            // Convert the selected gap edges to angular limits.
            // gapColumnToHeadingOffsetRad() accepts a continuous column
            // coordinate, therefore start-0.5 and end+0.5 represent the
            // physical left/right edges of the selected image corridor.
            const double edge_offset_a =
                gapColumnToHeadingOffsetRad(
                    static_cast<double>(selected_gap_start_col_) - 0.5);

            const double edge_offset_b =
                gapColumnToHeadingOffsetRad(
                    static_cast<double>(selected_gap_end_col_) + 0.5);

            double gap_min_heading_offset =
                std::min(edge_offset_a, edge_offset_b);
            double gap_max_heading_offset =
                std::max(edge_offset_a, edge_offset_b);

            // Keep a small angular margin from the visual gap boundaries.
            const double angular_edge_margin =
                std::abs(
                    gapColumnToHeadingOffsetRad(gap_edge_margin_cols_) -
                    gapColumnToHeadingOffsetRad(0.0));

            if ((gap_max_heading_offset - gap_min_heading_offset) >
                2.0 * angular_edge_margin)
            {
                gap_min_heading_offset += angular_edge_margin;
                gap_max_heading_offset -= angular_edge_margin;
            }

            // This is the goal direction projected into the safe corridor.
            const double safe_goal_heading_offset =
                clamp(
                    goal_heading_offset_body,
                    gap_min_heading_offset,
                    gap_max_heading_offset);

            const double safe_threshold =
                std::max(
                    0.01,
                    gap_safe_column_risk_threshold_);

            const double risk_fraction =
                clamp(
                    selected_gap_mean_risk_ /
                    safe_threshold,
                    0.0,
                    1.0);

            const double width_fraction =
                clamp(
                    static_cast<double>(selected_gap_width_cols_) /
                    static_cast<double>(
                        std::max(1, 2 * gap_min_width_cols_)),
                    0.0,
                    1.0);

            // Retain goal attraction during avoidance, but reduce its authority
            // as the selected corridor becomes riskier. Because the goal angle
            // was clamped into the selected gap above, this blend cannot steer
            // outside the visually accepted corridor.
            const double goal_heading_blend =
                clamp(
                    gap_goal_heading_blend_ *
                    (1.0 - 0.60 * risk_fraction),
                    0.0,
                    0.45);

            const double commanded_gap_heading_offset =
                (1.0 - goal_heading_blend) *
                    selected_gap_heading_offset_rad_ +
                goal_heading_blend *
                    safe_goal_heading_offset;

            const double gap_heading_world =
                normalizeAngle(
                    current_yaw_ +
                    commanded_gap_heading_offset);

            // Keep enough translation for reliable optical-flow expansion, but
            // lower speed inside narrow/riskier corridors. This is a smooth gap
            // mode speed policy, not a discontinuous TTC stop/go cap.
            const double risk_scale =
                1.0 - 0.35 * risk_fraction;

            const double width_scale =
                0.70 + 0.30 * width_fraction;

            const double nominal_gap_speed =
                clamp(
                    gap_max_forward_speed_ *
                        risk_scale *
                        width_scale,
                    gap_min_forward_speed_,
                    std::min(
                        gap_max_forward_speed_,
                        max_forward_speed_));

            // Gap guidance may choose WHERE to steer, but it must never undo
            // longitudinal braking.  'vx' already contains the APF/VPF/frontal
            // TTC safety reduction, so it is an upper bound on gap speed.
            const double gap_speed =
                std::min(
                    nominal_gap_speed,
                    std::max(0.0, vx));

            desired_vx_world =
                gap_speed *
                std::cos(gap_heading_world);

            desired_vy_world =
                gap_speed *
                std::sin(gap_heading_world);

            ROS_INFO_THROTTLE(
                0.35,
                "GAP_CMD: heading_world=%.1f deg centre_offset=%.1f deg cmd_offset=%.1f deg goal_offset=%.1f deg blend=%.2f speed=%.2f m/s desired=(%.2f, %.2f) gap=%d..%d risk=%.3f",
                gap_heading_world * 180.0 / M_PI,
                selected_gap_heading_offset_rad_ * 180.0 / M_PI,
                commanded_gap_heading_offset * 180.0 / M_PI,
                safe_goal_heading_offset * 180.0 / M_PI,
                goal_heading_blend,
                gap_speed,
                desired_vx_world,
                desired_vy_world,
                selected_gap_start_col_,
                selected_gap_end_col_,
                selected_gap_mean_risk_);
        }

        // ------------------------------------------------------
        // Overshoot recovery.
        // ------------------------------------------------------
        // Detect whether the UAV has passed the plane through the goal that is
        // perpendicular to the original start->goal path. When no immediate
        // danger is active, command a slow vector directly back toward the true
        // goal. This prevents the previous behaviour where the vehicle became
        // trapped on a vertical line after passing goal_x_.
        const double mission_path_dx =
            goal_x_ - start_x_;
        const double mission_path_dy =
            goal_y_ - start_y_;
        const double mission_path_norm =
            std::hypot(
                mission_path_dx,
                mission_path_dy);

        bool passed_goal_plane = false;
        if (mission_path_norm > 1e-6)
        {
            const double goal_plane_projection =
                (x - goal_x_) *
                    (mission_path_dx / mission_path_norm) +
                (y - goal_y_) *
                    (mission_path_dy / mission_path_norm);

            passed_goal_plane =
                goal_plane_projection > 0.0;
        }

        if (passed_goal_plane &&
            distance_to_goal > goal_tolerance_ &&
            !danger_active &&
            !rotational_apf_active_)
        {
            const double return_speed =
                std::min(
                    1.0,
                    std::max(
                        0.30,
                        0.60 * distance_to_goal));

            desired_vx_world =
                return_speed *
                (goal_x_ - x) /
                std::max(1e-6, distance_to_goal);

            desired_vy_world =
                return_speed *
                (goal_y_ - y) /
                std::max(1e-6, distance_to_goal);
        }

        double vx_world = desired_vx_world;
        double vy_world = desired_vy_world;
        double yaw_rate_cmd = 0.0;

        applyVelocityHeadingControl(
            desired_vx_world,
            desired_vy_world,
            vx_world,
            vy_world,
            yaw_rate_cmd);

        ROS_INFO_THROTTLE(
            0.5,
            "HEADING_CTRL: desired=(%.2f, %.2f) final=(%.2f, %.2f) yaw=%.1f deg desired_yaw=%.1f deg error=%.1f deg scale=%.2f body_v=(%.2f, %.2f) foe_valid=%d",
            desired_vx_world,
            desired_vy_world,
            vx_world,
            vy_world,
            current_yaw_ * 180.0 / M_PI,
            desired_heading_ * 180.0 / M_PI,
            last_heading_error_ * 180.0 / M_PI,
            last_motion_alignment_scale_,
            current_body_vx_,
            current_body_vy_,
            static_cast<int>(centre_foe_model_valid_));

        updateOscillationDetector(
            vy_world,
            x,
            y,
            distance_to_goal,
            danger_active);
        if (oscillation_failure_)
        {
            publishVelocity(0.0, 0.0, 0.0);
            ROS_ERROR("Terminating run because persistent oscillation was detected.");
            finishMission("Persistent oscillation", false);
            return;
        }

        const double error_z = goal_z_ - z;
        const double vz = clamp(k_z_ * error_z, -max_vertical_speed_, max_vertical_speed_);

        last_vx_ = vx_world;
        last_vy_ = vy_world;
        last_vz_ = vz;

        publishVelocity(vx_world, vy_world, vz, yaw_rate_cmd);
    }

    void finishMission(const std::string& termination_reason, bool success)
    {
        if (attempt_finished_ || experiment_complete_) { return; }
        attempt_finished_ = true;

        if (mission_timer_started_ && mission_end_time_ <= 0.0)
        {
            mission_end_time_ = ros::Time::now().toSec();
        }

        publishVelocity(0.0, 0.0, 0.0);
        last_vx_ = 0.0; last_vy_ = 0.0; last_vz_ = 0.0;

        completed_runs_++;
        if (success) { successful_runs_++; }

        ROS_INFO("========== RUN RESULTS ==========");
        ROS_INFO("Run: %d / %d", current_run_, target_runs_);
        ROS_INFO("Completed runs: %d", completed_runs_);
        ROS_INFO("Successful runs: %d", successful_runs_);
        ROS_INFO("Termination: %s", termination_reason.c_str());
        ROS_INFO("Success: %s", boolText(success).c_str());

        if (have_pose_)
        {
            ROS_INFO("Final Position: x=%.2f y=%.2f z=%.2f",
                     current_pose_.position.x, current_pose_.position.y, current_pose_.position.z);
        }

        ROS_INFO("Path Length: %.2f m", path_length_);
        ROS_INFO("Execution Time: %.2f s", getMissionExecutionTime());
        ROS_INFO("Collisions: %d", collision_count_);
        printClearanceMetrics();
        ROS_INFO("Total Oscillation Count: %d", total_oscillation_count_);
        ROS_INFO("Rotational APF used: %s", boolText(rotational_apf_used_).c_str());
        ROS_INFO("Rotational APF activations: %d", rotational_activation_count_);
        ROS_INFO("=================================");

        if (have_pose_)
        {
            appendExperimentResults(termination_reason, current_pose_.position.x,
                                    current_pose_.position.y, current_pose_.position.z, success);
        }
        else
        {
            appendExperimentResults(termination_reason, 0.0, 0.0, 0.0, success);
        }

        closePathLoggingFiles();

        if (completed_runs_ >= target_runs_)
        {
            experiment_complete_ = true;
            results_printed_ = true;
            ROS_INFO("========== EXPERIMENT COMPLETE ==========");
            ROS_INFO("Completed %d requested runs.", completed_runs_);
            ROS_INFO("Successful runs: %d / %d", successful_runs_, completed_runs_);
            ROS_INFO("Results appended to %s", results_file_path_.c_str());
            ROS_INFO("=========================================");
            ros::shutdown();
            return;
        }

        ROS_INFO("Scheduling Gazebo reset before run %d...", current_run_ + 1);
        next_run_timer_ = nh_public_.createTimer(ros::Duration(reset_delay_s_),
                                                 &OnlineLKAPFVelocity::resetWorldAndStartNextRun, this, true);
    }

    void resetWorldAndStartNextRun(const ros::TimerEvent&)
    {
        if (experiment_complete_) { return; }

        resetting_world_ = true;
        publishVelocity(0.0, 0.0, 0.0);

        std_srvs::Empty reset_srv;
        if (!gazebo_reset_client_.waitForExistence(ros::Duration(10.0)))
        {
            ROS_ERROR("Gazebo reset service %s is unavailable.", gazebo_reset_service_.c_str());
            experiment_complete_ = true;
            results_printed_ = true;
            ros::shutdown();
            return;
        }
        if (!gazebo_reset_client_.call(reset_srv))
        {
            ROS_ERROR("Failed to reset Gazebo world using %s.", gazebo_reset_service_.c_str());
            experiment_complete_ = true;
            results_printed_ = true;
            ros::shutdown();
            return;
        }

        current_run_++;
        resetStateForNextRun();
        openPathLoggingFiles(current_run_);
        enableMotors();

        resetting_world_ = false;
        attempt_finished_ = false;
        stabilizing_after_reset_ = true;
        stabilization_end_time_ = ros::Time::now().toSec() + stabilization_time_s_;

        ROS_INFO("Gazebo reset successful. Stabilizing for %.1f s before run %d / %d.",
                 stabilization_time_s_, current_run_, target_runs_);
    }

    void resetStateForNextRun()
    {
        stabilizing_after_reset_ = false;
        stabilization_end_time_ = 0.0;
        reached_goal_ = false;
        takeoff_done_ = false;
        results_printed_ = false;
        collision_count_ = 0;
        in_collision_ = false;

        path_length_ = 0.0;
        mission_start_time_ = 0.0;
        mission_end_time_ = 0.0;
        mission_timer_started_ = false;
        have_last_path_pose_ = false;
        last_path_x_ = 0.0; last_path_y_ = 0.0; last_path_z_ = 0.0;

        current_obstacle_clearance_ = std::numeric_limits<double>::max();
        minimum_obstacle_clearance_ = std::numeric_limits<double>::max();
        minimum_collision_free_clearance_ = std::numeric_limits<double>::max();
        minimum_clearance_obstacle_x_ = 0.0;
        minimum_clearance_obstacle_y_ = 0.0;
        minimum_safe_clearance_obstacle_x_ = 0.0;
        minimum_safe_clearance_obstacle_y_ = 0.0;

        have_prev_image_ = false;
        prev_gray_.release();
        prev_pts_.clear();
        prev_time_ = 0.0;
        have_ttc_filtered_ = false;
        ttc_filtered_ = 0.0;
        ttc_miss_count_ = 0;
        have_frontal_ttc_filtered_ = false;
        frontal_ttc_filtered_ = 0.0;
        frontal_ttc_miss_count_ = 0;
        vpf_risk_ = cv::Mat::zeros(vpf_grid_rows_, vpf_grid_cols_, CV_32F);
        repulsive_y_ = 0.0;
        repulsive_y_filtered_ = 0.0;
        left_danger_ = 0.0;
        right_danger_ = 0.0;
        tracked_features_ = 0;
        resetGapGuidanceState();

        current_body_vx_ = 0.0;
        current_body_vy_ = 0.0;
        current_body_vz_ = 0.0;
        desired_heading_ = 0.0;
        last_heading_error_ = 0.0;
        last_motion_alignment_scale_ = 1.0;
        centre_foe_model_valid_ = false;
        last_centre_foe_valid_time_ = 0.0;
        have_last_centre_foe_valid_time_ = false;
        dynamic_foe_valid_ = false;
        dynamic_foe_x_px_ = 320.0;
        dynamic_foe_y_px_ = 240.0;
        frontal_corridor_x_min_px_ = 0.0;
        frontal_corridor_x_max_px_ = 0.0;

        vx_command_ = 0.0;
        vy_filtered_ = 0.0;
        longitudinal_frontal_risk_filtered_ = 0.0;
        last_forward_attraction_target_ = 0.0;
        last_longitudinal_frontal_risk_ = 0.0;
        last_longitudinal_repulsive_vx_ = 0.0;
        last_forward_resultant_target_ = 0.0;
        last_avoid_direction_ = 1.0;
        last_vx_ = 0.0; last_vy_ = 0.0; last_vz_ = 0.0;

        rotational_apf_active_ = false;
        rotational_apf_used_ = false;
        rotational_direction_ = 0;
        rotational_activation_count_ = 0;
        rotational_last_request_time_ = 0.0;
        rotational_blockage_active_ = false;
        rotational_blockage_start_time_ = 0.0;
        last_rotational_blockage_duration_ = 0.0;
        rotational_safe_gap_reacquire_active_ = false;
        rotational_safe_gap_reacquire_start_time_ = 0.0;
        last_visual_obstacle_x_body_ = 1.0;
        last_visual_obstacle_y_body_ = 0.0;
        last_rotational_risk_strength_ = 0.0;
        last_rotational_vx_body_ = 0.0;
        last_rotational_vy_body_ = 0.0;
        rotational_vx_filtered_ = 0.0;
        rotational_vy_filtered_ = 0.0;

        last_y_ref_ = 0.0;
        last_cross_track_error_ = 0.0;
        last_v_path_y_ = 0.0;
        last_v_goal_y_ = 0.0;

        clearOscillationEpisode();
        total_oscillation_count_ = 0;
        oscillation_failure_ = false;

        progress_poor_ = false;
        progress_rate_ = 0.0;
        prev_distance_to_goal_ = 0.0;
        progress_timer_start_ = 0.0;
        have_prev_distance_to_goal_ = false;
        have_progress_timer_start_ = false;

        movement_watch_started_ = false;
        last_movement_time_ = 0.0;
        movement_anchor_x_ = 0.0;
        movement_anchor_y_ = 0.0;

        resetPerformanceMetrics();
        have_pose_ = false;
    }

    void printClearanceMetrics() const
    {
        if (minimum_obstacle_clearance_ < std::numeric_limits<double>::max())
        {
            ROS_INFO("Minimum Clearance Including Collisions: %.3f m", minimum_obstacle_clearance_);
            ROS_INFO("Overall Closest Obstacle: x=%.2f y=%.2f",
                     minimum_clearance_obstacle_x_, minimum_clearance_obstacle_y_);
        }
        else
        {
            ROS_INFO("Minimum Clearance Including Collisions: N/A");
        }

        if (minimum_collision_free_clearance_ < std::numeric_limits<double>::max())
        {
            ROS_INFO("Minimum Collision-Free Clearance: %.3f m", minimum_collision_free_clearance_);
            ROS_INFO("Closest Obstacle Without Collision: x=%.2f y=%.2f",
                     minimum_safe_clearance_obstacle_x_, minimum_safe_clearance_obstacle_y_);
        }
        else
        {
            ROS_INFO("Minimum Collision-Free Clearance: N/A");
        }
    }

    void printPerformanceMetrics()
    {
        ROS_INFO("Image Count: %d", image_count_);
        ROS_INFO("Average Image Rate: %.2f Hz", getAverageImageRate());
        if (image_count_ > 1 && min_image_dt_ < std::numeric_limits<double>::max())
        {
            ROS_INFO("Minimum Image Interval: %.3f s", min_image_dt_);
            ROS_INFO("Maximum Image Interval: %.3f s", max_image_dt_);
        }
        ROS_INFO("Average Processing Time: %.2f ms", getAverageProcessingTimeMs());
        ROS_INFO("Maximum Processing Time: %.2f ms", max_processing_time_ms_);
    }

    void printFinalResults()
    {
        if (results_printed_ || experiment_complete_) { return; }
        if (mission_timer_started_ && mission_end_time_ <= 0.0)
        {
            mission_end_time_ = ros::Time::now().toSec();
        }
        results_printed_ = true;

        ROS_INFO("========== FINAL RESULTS ==========");
        ROS_INFO("Termination reason: Manual stop / node shutdown");
        if (have_pose_)
        {
            ROS_INFO("Final Position: x=%.2f y=%.2f z=%.2f",
                     current_pose_.position.x, current_pose_.position.y, current_pose_.position.z);
        }
        ROS_INFO("Path Length: %.2f m", path_length_);
        ROS_INFO("Execution Time: %.2f s", getMissionExecutionTime());
        ROS_INFO("Tracked Features: %d", tracked_features_);
        ROS_INFO("Total Oscillation Count: %d", total_oscillation_count_);
        ROS_INFO("Collisions: %d", collision_count_);
        printClearanceMetrics();
        printPerformanceMetrics();
        ROS_INFO("===================================");

        if (have_pose_)
        {
            appendExperimentResults("Manual stop / node shutdown", current_pose_.position.x,
                                    current_pose_.position.y, current_pose_.position.z, false);
        }
        else
        {
            appendExperimentResults("Manual stop / node shutdown", 0.0, 0.0, 0.0, false);
        }
        closePathLoggingFiles();
    }

private:
    double getMissionExecutionTime() const
    {
        if (!mission_timer_started_) { return 0.0; }
        const double end_time = (mission_end_time_ > 0.0) ? mission_end_time_ : ros::Time::now().toSec();
        return std::max(0.0, end_time - mission_start_time_);
    }

    static double normalizeAngle(double angle)
    {
        while (angle > M_PI) { angle -= 2.0 * M_PI; }
        while (angle < -M_PI) { angle += 2.0 * M_PI; }
        return angle;
    }

    static double yawFromQuaternion(const geometry_msgs::Quaternion& q)
    {
        const double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
        const double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
        return std::atan2(siny_cosp, cosy_cosp);
    }

    static double clamp(double value, double low, double high)
    {
        return std::max(low, std::min(high, value));
    }

    static double sign(double value)
    {
        if (value > 0.0) return 1.0;
        if (value < 0.0) return -1.0;
        return 0.0;
    }

    static std::string boolText(bool value) { return value ? "True" : "False"; }

    static std::string formatString(const char* fmt, double value)
    {
        char buffer[128];
        std::snprintf(buffer, sizeof(buffer), fmt, value);
        return std::string(buffer);
    }

    static std::string formatString(const char* fmt, int value)
    {
        char buffer[128];
        std::snprintf(buffer, sizeof(buffer), fmt, value);
        return std::string(buffer);
    }

    ros::NodeHandle nh_public_;
    ros::NodeHandle nh_;
    ros::Publisher vel_pub_;
    ros::Publisher vis_pub_;
    ros::Subscriber image_sub_;
    ros::Subscriber pose_sub_;
    ros::Timer control_timer_;
    ros::Timer next_run_timer_;
    ros::ServiceClient gazebo_reset_client_;

    int target_runs_;
    int current_run_;
    int completed_runs_;
    int successful_runs_;
    bool attempt_finished_;
    bool resetting_world_;
    bool stabilizing_after_reset_;
    bool experiment_complete_;
    double stabilization_time_s_;
    double stabilization_end_time_;
    double no_movement_timeout_s_;
    double movement_threshold_m_;
    double reset_delay_s_;
    std::string gazebo_reset_service_;

    bool movement_watch_started_;
    double last_movement_time_;
    double movement_anchor_x_;
    double movement_anchor_y_;

    bool enable_visualization_;

    bool use_gaussian_blur_;
    int gaussian_kernel_size_;
    double gaussian_sigma_;

    double first_image_time_;
    double last_image_time_;
    double previous_image_time_for_dt_;
    int image_count_;
    double min_image_dt_;
    double max_image_dt_;

    double processing_time_sum_ms_;
    double max_processing_time_ms_;
    int processing_count_;

    double last_vx_, last_vy_, last_vz_;

    bool use_path_controller_;
    double k_path_y_;
    double k_goal_y_;
    double path_lookahead_distance_;
    double max_path_recovery_speed_;
    double last_y_ref_;
    double last_cross_track_error_;
    double last_v_path_y_;
    double last_v_goal_y_;

    double current_yaw_;
    double current_yaw_rate_;
    double yaw_rate_filter_alpha_;

    double k_yaw_recover_;
    double max_yaw_recover_rate_;

    bool enable_velocity_heading_control_;
    double k_velocity_heading_;
    double max_velocity_heading_rate_;
    double heading_min_horizontal_speed_;
    double heading_slow_angle_deg_;
    double heading_strong_slow_angle_deg_;
    double heading_slow_scale_;
    double heading_strong_slow_scale_;
    double max_body_lateral_speed_;

    bool enable_centre_foe_validity_gate_;
    double centre_foe_min_forward_body_speed_;
    double centre_foe_max_lateral_forward_ratio_;
    double centre_foe_max_heading_error_deg_;
    double centre_foe_ttc_hold_s_;
    double centre_foe_risk_decay_;

    double current_body_vx_;
    double current_body_vy_;
    double current_body_vz_;
    double desired_heading_;
    double last_heading_error_;
    double last_motion_alignment_scale_;
    bool centre_foe_model_valid_;
    double last_centre_foe_valid_time_;
    bool have_last_centre_foe_valid_time_;

    double dynamic_foe_min_forward_body_speed_;
    double dynamic_foe_max_offset_image_widths_;
    double dynamic_foe_x_px_;
    double dynamic_foe_y_px_;
    bool dynamic_foe_valid_;
    double frontal_corridor_x_min_px_;
    double frontal_corridor_x_max_px_;

    bool enable_yaw_flow_compensation_;
    double camera_fx_pixels_;
    double camera_fy_pixels_;
    double yaw_flow_sign_;
    double max_yaw_rate_for_compensation_;

    cv::Mat prev_gray_;
    std::vector<cv::Point2f> prev_pts_;
    double prev_time_;
    bool have_prev_image_;

    geometry_msgs::Pose current_pose_;
    bool have_pose_;

    double goal_tolerance_;
    bool reached_goal_;
    bool takeoff_done_;

    int collision_count_;
    bool in_collision_;
    bool results_printed_;

    double uav_collision_radius_;
    double obstacle_radius_;
    double current_obstacle_clearance_;
    double minimum_obstacle_clearance_;
    double minimum_collision_free_clearance_;
    double minimum_clearance_obstacle_x_;
    double minimum_clearance_obstacle_y_;
    double minimum_safe_clearance_obstacle_x_;
    double minimum_safe_clearance_obstacle_y_;

    std::ofstream path_file_;
    std::ofstream obstacle_file_;
    bool path_files_opened_;
    std::string results_file_path_;

    double path_length_;
    double mission_start_time_;
    double mission_end_time_;
    bool mission_timer_started_;
    bool have_last_path_pose_;
    double last_path_x_;
    double last_path_y_;
    double last_path_z_;

    std::vector<std::pair<double, double>> obstacles_;
    std::vector<std::pair<double, double>> obstacles_medium_forest;
    std::vector<std::pair<double, double>> obstacles_zigzag;
    std::vector<std::pair<double, double>> obstacles_u_shape_trap;

    double vy_sign_threshold_;
    double prev_vy_sign_;
    double minimum_lateral_reversal_distance_;
    double last_accepted_reversal_lateral_;
    bool have_last_accepted_reversal_lateral_;
    double oscillation_trap_radius_;
    double oscillation_minimum_goal_progress_;
    double oscillation_episode_start_x_;
    double oscillation_episode_start_y_;
    double oscillation_episode_start_distance_to_goal_;
    int oscillation_count_;
    int total_oscillation_count_;
    int oscillation_limit_;
    bool oscillation_failure_;
    double oscillation_failure_time_;
    bool oscillation_active_;
    double oscillation_memory_time_;
    double oscillation_episode_start_time_;
    double last_oscillation_time_;
    bool have_last_oscillation_time_;
    bool have_oscillation_episode_start_;

    bool progress_poor_;
    double progress_rate_;
    double prev_distance_to_goal_;
    double progress_timer_start_;
    bool have_prev_distance_to_goal_;
    bool have_progress_timer_start_;
    double progress_window_;
    double min_progress_rate_;

    double start_x_, start_y_;
    double goal_x_, goal_y_, goal_z_;

    double forward_speed_;
    double vx_command_;
    double max_forward_acceleration_;
    double max_forward_deceleration_;
    double max_forward_speed_;
    double max_lateral_speed_;
    double max_vertical_speed_;

    double default_avoid_y_speed_;
    double danger_ttc_;
    double last_avoid_direction_;
    double vy_filtered_;
    double vy_alpha_;

    double k_att_y_;
    double k_rep_y_;
    double k_z_;

    bool enable_longitudinal_repulsion_;
    double k_longitudinal_repulsion_;
    double longitudinal_repulsion_filter_alpha_;
    double longitudinal_frontal_exponent_;
    double longitudinal_frontal_risk_filtered_;
    double last_forward_attraction_target_;
    double last_longitudinal_frontal_risk_;
    double last_longitudinal_repulsive_vx_;
    double last_forward_resultant_target_;

    bool enable_rotational_apf_;
    bool rotational_trigger_on_trap_;
    bool rotational_apf_active_;
    bool rotational_apf_used_;
    int rotational_direction_;
    int rotational_activation_count_;

    double k_rotational_;
    double rotational_activation_risk_;
    double rotational_maintain_risk_;
    double rotational_start_forward_speed_;
    double rotational_start_delay_s_;
    double rotational_direction_deadband_rad_;
    double rotational_min_speed_;
    double rotational_max_speed_;
    double rotational_radial_escape_weight_;
    double rotational_forward_speed_cap_;
    double rotational_safe_gap_release_delay_s_;
    bool rotational_safe_gap_reacquire_active_;
    double rotational_safe_gap_reacquire_start_time_;
    double rotational_filter_alpha_;
    double rotational_release_delay_s_;
    double rotational_last_request_time_;
    bool rotational_blockage_active_;
    double rotational_blockage_start_time_;
    double last_rotational_blockage_duration_;

    double last_visual_obstacle_x_body_;
    double last_visual_obstacle_y_body_;
    double last_rotational_risk_strength_;
    double last_rotational_vx_body_;
    double last_rotational_vy_body_;
    double rotational_vx_filtered_;
    double rotational_vy_filtered_;

    int last_image_width_;
    int last_image_height_;

    int vpf_grid_cols_;
    int vpf_grid_rows_;
    cv::Mat vpf_risk_;
    double vpf_ttc_weight_;
    double vpf_flow_weight_;

    bool enable_gap_guidance_;
    double gap_safe_column_risk_threshold_;
    double gap_activation_risk_threshold_;
    double gap_rotational_trigger_risk_threshold_;
    int gap_min_width_cols_;
    double gap_hold_time_s_;
    double gap_release_delay_s_;
    double gap_no_safe_latch_s_;
    double gap_goal_bias_weight_;
    double gap_goal_heading_blend_;
    double avoidance_attraction_scale_;
    double rotational_attraction_scale_;
    double gap_continuity_weight_;
    double gap_width_bonus_weight_;
    double gap_max_forward_speed_;
    double gap_min_forward_speed_;
    double gap_edge_margin_cols_;

    std::vector<double> column_risk_;
    bool gap_guidance_active_;
    bool safe_gap_available_;
    bool gap_escape_required_;
    int selected_gap_start_col_;
    int selected_gap_end_col_;
    double selected_gap_center_col_;
    int selected_gap_width_cols_;
    double selected_gap_heading_offset_rad_;
    double selected_gap_mean_risk_;
    double selected_gap_score_;
    double max_column_risk_;
    double last_gap_selection_time_;
    double last_gap_obstacle_time_;
    bool have_last_gap_obstacle_time_;

    // Persistent gap-side commitment and short no-gap activation latch.
    int committed_gap_side_;
    double last_committed_gap_time_;
    double last_no_safe_gap_time_;
    bool have_last_no_safe_gap_time_;

    int risk_shaping_mode_;
    double inverse_ttc_epsilon_;
    double risk_power_alpha_;
    double logistic_beta_;
    double logistic_threshold_;
    double max_ttc_risk_;

    double repulsive_y_;
    double repulsive_y_filtered_;
    double repulsive_alpha_;
    double left_danger_;
    double right_danger_;
    int tracked_features_;
    double ttc_filtered_;
    bool have_ttc_filtered_;
    double ttc_max_useful_;
    int ttc_miss_count_;
    int ttc_miss_limit_;

    double frontal_ttc_filtered_;
    bool have_frontal_ttc_filtered_;
    int frontal_ttc_miss_count_;
    int frontal_ttc_miss_limit_;
    double frontal_ttc_danger_;
    double frontal_ttc_min_x_fraction_;
    double frontal_ttc_max_x_fraction_;
    double frontal_ttc_quantile_;
    double frontal_ttc_filter_alpha_;
    int frontal_ttc_min_features_;

    double floor_row_min_frac_;
    double floor_row_max_frac_;
    double max_lk_error_;
    double max_flow_pixels_;

    double minimum_vpf_flow_pixels_;
    double minimum_radial_ratio_;
    double local_danger_ttc_;
    int minimum_cell_support_;

    int max_corners_;
    double quality_level_;
    double min_distance_;
    int block_size_;
    cv::Size lk_win_size_;
    int lk_max_level_;
    cv::TermCriteria lk_criteria_;
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "online_lk_apf_velocity_avoidance");
    OnlineLKAPFVelocity node;
    ros::spin();
    node.printFinalResults();
    return 0;
}