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
// This is a from-scratch cleanup of the previous node. Scope for this pass:
//   - Correct, minimal LK + VPF + APF pipeline.
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
        k_goal_y_ = 0.08;   // small direct pull toward the final goal_y
        last_y_ref_ = 0.0;
        last_cross_track_error_ = 0.0;
        last_v_path_y_ = 0.0;
        last_v_goal_y_ = 0.0;

        // Heading recovery keeps the camera approximately forward-facing.
        // The measured yaw rate is also used to remove rotation-induced
        // optical flow before TTC and VPF calculations.
        current_yaw_ = 0.0;
        current_yaw_rate_ = 0.0;
        yaw_rate_filter_alpha_ = 0.25;

        k_yaw_recover_ = 0.6;
        max_yaw_recover_rate_ = 0.3;

        // Camera model for yaw-flow compensation. The defaults match a
        // 640-pixel-wide camera with approximately 90 degree horizontal FOV.
        // They can be overridden from rosrun without editing the source.
        nh_.param("enable_yaw_flow_compensation",
                  enable_yaw_flow_compensation_,
                  true);
        nh_.param("camera_fx_pixels",
                  camera_fx_pixels_,
                  320.0);
        nh_.param("camera_fy_pixels",
                  camera_fy_pixels_,
                  320.0);
        nh_.param("yaw_flow_sign",
                  yaw_flow_sign_,
                  -1.0);
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
        obstacles_ = obstacles_front_wall;

        openPathLoggingFiles(current_run_);

        // ------------------------------------------------------
        // Oscillation detector - SAFETY NET ONLY in this version.
        // It no longer drives any control decision (no rotational-APF
        // trigger, no escape controller). It just terminates a run if the
        // commanded lateral velocity keeps reversing sign for too long,
        // which is a useful "something is wrong, stop and log it" signal.
        // ------------------------------------------------------
        vy_sign_threshold_ = 0.2;
        prev_vy_sign_ = 0.0;
        oscillation_count_ = 0;
        total_oscillation_count_ = 0;
        oscillation_limit_ = 4;
        oscillation_failure_ = false;
        oscillation_failure_time_ = 8.0; //NURA: 20.0;
        oscillation_active_ = false;
        oscillation_memory_time_ = 2.0; //NURA: 5.0;
        oscillation_episode_start_time_ = 0.0;
        last_oscillation_time_ = 0.0;

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
        // start_y_ = 2;
        // goal_x_ = 8.0;
        // goal_y_ = 9.5;
        
        //U SHAPE
        start_x_ = -2.0;//-2.0;
        start_y_ = 6.0;//5.0;
        goal_x_ = 14.0; //12.0;
        goal_y_ = 6; //5.0;
        
        //TWO TREES
        // start_x_ = 0;
        // start_y_ = 6;
        // goal_x_ = 12.0;
        // goal_y_ = 6;

        goal_z_ = 1.50;

        // Forward velocity settings [m/s]
        forward_speed_ = 10.0; //sparse 10
        vx_command_ = 0.0;
        max_forward_acceleration_ = 2;
        max_forward_speed_ = 25.0;
        max_lateral_speed_ = 10.0; //sparse 10
        max_vertical_speed_ = 0.5;

        // Fallback lateral nudge when danger is centred in the image (neither
        // side clearly worse) - keeps committing to the last meaningful side
        // instead of dithering near zero.
        default_avoid_y_speed_ = 0.8; //0.8
        danger_ttc_ = 5.0;
        last_avoid_direction_ = 1; //1

        // Final lateral command smoothing. Kept light - the heavy lifting
        // against noise now happens on repulsive_y_ itself (see below), so
        // this only needs to remove residual per-tick jitter.
        vy_filtered_ = 0.0;
        vy_alpha_ = 0.4;

        // APF gains
        k_att_y_ = 0.8;   // used only if use_path_controller_ is false
        k_rep_y_ = 2;
        k_z_ = 1.0;

        // ------------------------------------------------------
        // Rotational APF: true tangential force in the horizontal X-Y plane.
        //
        // The ordinary controller still supplies:
        //   - forward/path-following motion,
        //   - TTC-based longitudinal slowdown,
        //   - lateral VPF repulsion.
        //
        // The rotational term estimates the dominant obstacle bearing from
        // the active VPF cells, forms the obstacle-away radial unit vector,
        // and rotates that vector by +/-90 degrees. This produces both a
        // forward/backward and a lateral correction rather than a fixed
        // lateral "escape nudge".
        //
        // Set _enable_rotational_apf:=false for the baseline comparison.
        // Set _rotational_trigger_on_trap:=true to activate only after poor
        // progress or sustained oscillation; false activates whenever the
        // visual repulsive field is active (closer to the published method).
        // ------------------------------------------------------
        nh_.param("enable_rotational_apf",
                  enable_rotational_apf_,
                  true);
        nh_.param("rotational_trigger_on_trap",
                  rotational_trigger_on_trap_,
                  false);
        nh_.param("k_rotational",
                  k_rotational_,
                  2.0);
        nh_.param("rotational_activation_risk",
                  rotational_activation_risk_,
                  0.20);
        nh_.param("rotational_min_speed",
                  rotational_min_speed_,
                  0.35);
        nh_.param("rotational_max_speed",
                  rotational_max_speed_,
                  1.50);
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
        // Trusted image band for the first controlled VPF test.
        // ------------------------------------------------------
        floor_row_min_frac_ = 0.35;
        floor_row_max_frac_ = 0.80;

        // LK track validation.
        max_lk_error_ = 20.0;
        max_flow_pixels_ = 40.0;

        // Simplified centre-FOE VPF test parameters.
        minimum_vpf_flow_pixels_ = 0.20;
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
        ROS_INFO("Heading recovery enabled: yaw regulated toward zero.");
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
            << "vx,vy,vz,ttc,current_obstacle_clearance_m,"
            << "repulsive_y_raw,repulsive_y_filtered,left_danger,right_danger,"
            << "rotational_active,rotational_direction,"
            << "rotational_vx_body,rotational_vy_body,rotational_risk_strength,"
            << "y_path_ref,cross_track_error,v_path_y,v_goal_y,"
            << "tracked_features,progress_poor,collision_count\n";

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
                << "max_lateral_speed_mps,"
                << "rotational_apf_enabled,rotational_apf_used,rotational_activation_count,"
                << "rotational_trigger_on_trap,k_rotational,"
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
            << boolText(enable_rotational_apf_) << ","
            << boolText(rotational_apf_used_) << ","
            << rotational_activation_count_ << ","
            << boolText(rotational_trigger_on_trap_) << ","
            << k_rotational_ << ","
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
        path_file_ << ",";

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
            << last_y_ref_ << "," << last_cross_track_error_ << ","
            << last_v_path_y_ << "," << last_v_goal_y_ << ","
            << tracked_features_ << "," << static_cast<int>(progress_poor_) << ","
            << collision_count_ << "\n";
    }

    void poseCallback(const nav_msgs::OdometryConstPtr& msg)
    {
        current_pose_ = msg->pose.pose;
        current_yaw_ = yawFromQuaternion(current_pose_.orientation);

        // nav_msgs/Odometry expresses twist in the child/body frame.
        // Positive angular.z is the measured body yaw rate.
        const double measured_yaw_rate =
            msg->twist.twist.angular.z;

        current_yaw_rate_ =
            yaw_rate_filter_alpha_ * measured_yaw_rate +
            (1.0 - yaw_rate_filter_alpha_) * current_yaw_rate_;

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

        double ttc = 0.0;
        const bool ttc_valid = estimateTTC(frame, good_old, good_new, gray.size(), dt, ttc);

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
                ttc_filtered_ = alpha * ttc + (1.0 - alpha) * ttc_filtered_;
            }
        }
        else
        {
            ttc_miss_count_++;
            if (ttc_miss_count_ > ttc_miss_limit_) { have_ttc_filtered_ = false; }
        }

        vpf_risk_ = computeVisualPotentialField(frame, good_old, good_new, gray.size(), dt);
        computeVPFRepulsionY(repulsive_y_, left_danger_, right_danger_);

        // Smooth the raw repulsion signal here, at the source, before any
        // gain or direction-hysteresis logic ever sees it.
        repulsive_y_filtered_ =
            repulsive_alpha_ * repulsive_y_ + (1.0 - repulsive_alpha_) * repulsive_y_filtered_;

        if (std::abs(repulsive_y_filtered_) > 0.25)
        {
            last_avoid_direction_ = sign(repulsive_y_filtered_);
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
        double dt)
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

        // After measured yaw-flow compensation, the remaining translation is
        // assumed to be forward-dominant, so the FOE is approximated by the
        // image centre for this controlled test.
        const double foe_x = 0.5 * static_cast<double>(w);
        const double foe_y = 0.5 * static_cast<double>(h);
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

            if (radius < 20.0)
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

    // Safety-net only: detects sustained lateral-command sign reversals and
    // ends the run if it persists too long. Does not influence any control
    // decision - purely diagnostic/termination.
    void updateOscillationDetector(double vy)
    {
        const double now = ros::Time::now().toSec();
        double current_sign = 0.0;
        if (vy >= vy_sign_threshold_) { current_sign = 1.0; }
        else if (vy <= -vy_sign_threshold_) { current_sign = -1.0; }

        if (current_sign == 0.0)
        {
            if (have_last_oscillation_time_ && (now - last_oscillation_time_) > oscillation_memory_time_)
            {
                oscillation_count_ = 0;
                oscillation_active_ = false;
                have_oscillation_episode_start_ = false;
                have_last_oscillation_time_ = false;
                prev_vy_sign_ = 0.0;
            }
            return;
        }

        if (prev_vy_sign_ == 0.0)
        {
            prev_vy_sign_ = current_sign;
            return;
        }

        if (current_sign != prev_vy_sign_)
        {
            oscillation_count_++;
            total_oscillation_count_++;
            last_oscillation_time_ = now;
            have_last_oscillation_time_ = true;

            if (!have_oscillation_episode_start_)
            {
                oscillation_episode_start_time_ = now;
                have_oscillation_episode_start_ = true;
                ROS_WARN("Oscillation episode started.");
            }
            if (oscillation_count_ >= oscillation_limit_) { oscillation_active_ = true; }
            prev_vy_sign_ = current_sign;
        }

        if (have_last_oscillation_time_ && (now - last_oscillation_time_) > oscillation_memory_time_)
        {
            ROS_INFO("Oscillation episode cleared after %.2f s without a reversal.", now - last_oscillation_time_);
            oscillation_count_ = 0;
            oscillation_active_ = false;
            have_oscillation_episode_start_ = false;
            have_last_oscillation_time_ = false;
            prev_vy_sign_ = current_sign;
            return;
        }

        if (oscillation_active_ && have_oscillation_episode_start_ && have_last_oscillation_time_)
        {
            const double episode_duration = now - oscillation_episode_start_time_;
            const double time_since_last_reversal = now - last_oscillation_time_;
            const bool reversals_are_still_recent = time_since_last_reversal <= oscillation_memory_time_;

            if (episode_duration >= oscillation_failure_time_ && reversals_are_still_recent)
            {
                oscillation_failure_ = true;
                ROS_ERROR("Persistent oscillation detected: duration=%.2f s, count=%d, total=%d.",
                          episode_duration, oscillation_count_, total_oscillation_count_);
            }
        }
    }

    bool estimateTTC(const cv::Mat& color_frame,
                     const std::vector<cv::Point2f>& old_pts,
                     const std::vector<cv::Point2f>& new_pts,
                     const cv::Size& image_size,
                     double dt,
                     double& ttc_out)
    {
        const int w = image_size.width;
        const int h = image_size.height;
        const double cx = w / 2.0;
        const double cy = h / 2.0;
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

            const double rx = x_old - cx;
            const double ry = y_old - cy;

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

            if (r < 20.0)
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
    double desiredYOnPath(double x) const
    {
        const double dx = goal_x_ - start_x_;
        if (std::abs(dx) < 1e-6) { return goal_y_; }
        const double s = clamp((x - start_x_) / dx, 0.0, 1.0);
        return start_y_ + s * (goal_y_ - start_y_);
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

    // Activate/deactivate the rotational APF and select one rotation direction.
    //
    // The direction rule follows the planned-path/obstacle-bearing principle:
    // theta = phi - rho. For theta >= 0 use the clockwise matrix
    // [0 1; -1 0]; otherwise use the counter-clockwise matrix
    // [0 -1; 1 0]. The selected direction is locked while the current
    // avoidance episode is active to prevent noisy VPF centroid estimates from
    // reversing the tangential force every frame.
    void updateRotationalAPFState(
        double x,
        double y,
        bool danger_active,
        double now)
    {
        double obstacle_x_body = 1.0;
        double obstacle_y_body = 0.0;
        double risk_strength = 0.0;

        const bool visual_obstacle_valid =
            estimateDominantVisualObstacleBody(
                obstacle_x_body,
                obstacle_y_body,
                risk_strength);

        if (visual_obstacle_valid)
        {
            last_visual_obstacle_x_body_ =
                obstacle_x_body;

            last_visual_obstacle_y_body_ =
                obstacle_y_body;

            last_rotational_risk_strength_ =
                risk_strength;
        }

        const bool trap_trigger_satisfied =
            !rotational_trigger_on_trap_ ||
            oscillation_active_ ||
            progress_poor_;

        const bool rotational_requested =
            enable_rotational_apf_ &&
            danger_active &&
            visual_obstacle_valid &&
            risk_strength >= rotational_activation_risk_ &&
            trap_trigger_satisfied;

        if (rotational_requested)
        {
            rotational_last_request_time_ = now;

            if (!rotational_apf_active_)
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

                // +1 denotes clockwise R=[0 1; -1 0].
                rotational_direction_ =
                    (theta >= 0.0) ? 1 : -1;

                rotational_apf_active_ = true;
                rotational_apf_used_ = true;
                rotational_activation_count_++;

                ROS_WARN(
                    "Rotational APF activated: direction=%s, theta=%.3f rad, risk=%.3f.",
                    rotational_direction_ > 0 ? "CW" : "CCW",
                    theta,
                    risk_strength);
            }

            return;
        }

        if (rotational_apf_active_ &&
            (now - rotational_last_request_time_) >=
                rotational_release_delay_s_)
        {
            ROS_INFO(
                "Rotational APF released after obstacle influence decreased.");

            rotational_apf_active_ = false;
            rotational_direction_ = 0;
        }
    }

    // Compute the +/-90-degree tangential vector in the BODY X-Y plane.
    //
    // radial = UAV - obstacle (points away from obstacle)
    // clockwise tangent        = [ radial_y, -radial_x ]
    // counter-clockwise tangent= [-radial_y,  radial_x ]
    //
    // Both components are returned. Therefore the rotational term can create
    // forward, lateral, or backward motion as the obstacle bearing changes.
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

            if (tangent_norm > 1e-6)
            {
                tangent_x_body /= tangent_norm;
                tangent_y_body /= tangent_norm;

                const double rotational_speed =
                    clamp(
                        k_rotational_ * current_risk,
                        rotational_min_speed_,
                        rotational_max_speed_);

                target_vx_body =
                    rotational_speed *
                    tangent_x_body;

                target_vy_body =
                    rotational_speed *
                    tangent_y_body;
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
            repulsive_y_ = 0.0;
            repulsive_y_filtered_ = 0.0;
            left_danger_ = 0.0;
            right_danger_ = 0.0;
            tracked_features_ = 0;
            current_yaw_rate_ = 0.0;
            vx_command_ = 0.0;
            vy_filtered_ = 0.0;

            rotational_apf_active_ = false;
            rotational_direction_ = 0;
            rotational_last_request_time_ = 0.0;
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
                resetPerformanceMetrics();
            }
            return;
        }

        const double distance_to_goal =
            std::sqrt((goal_x_ - x) * (goal_x_ - x) + (goal_y_ - y) * (goal_y_ - y));
        const bool danger_active = have_ttc_filtered_ && (ttc_filtered_ < danger_ttc_);

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
        updateRotationalAPFState(x, y, danger_active, now);

        const bool passed_goal_x = x >= goal_x_;
        const bool y_close = std::abs(goal_y_ - y) <= 1.0;
        if (distance_to_goal < goal_tolerance_ || (passed_goal_x && y_close)) { reached_goal_ = true; }

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
        // Forward speed schedule + TTC-based slowdown
        // ------------------------------------------------------
        double vx = forward_speed_;
        if (distance_to_goal < 8.0) { vx = std::min(vx, 5.0); }
        if (distance_to_goal < 4.0) { vx = std::min(vx, 3.0); }
        if (distance_to_goal < 2.0) { vx = std::min(vx, 1.0); }
        if (distance_to_goal < 0.5) { vx = 0.3; }
        if (x >= goal_x_) { vx = 0.0; }


        if (danger_active)
        {
            if (ttc_filtered_ < 3.0) { vx *= 0.2; }
            else if (ttc_filtered_ < 5.0) { vx *= 0.5; }
            else if (ttc_filtered_ < danger_ttc_) { vx *= 0.8; }
        }

        vx = clamp(vx, 0.0, max_forward_speed_);

        const double dt = 0.01;
        const double max_delta = max_forward_acceleration_ * dt;
        if (vx > vx_command_) { vx_command_ = std::min(vx_command_ + max_delta, vx); }
        else { vx_command_ = std::max(vx_command_ - max_delta, vx); }
        vx = vx_command_;

        // ------------------------------------------------------
        // APF attraction term (world-frame, path-following)
        // ------------------------------------------------------
        double v_att_y = 0.0;
        if (use_path_controller_)
        {
            last_y_ref_ = desiredYOnPath(x);
            last_cross_track_error_ = last_y_ref_ - y;
            last_v_path_y_ = k_path_y_ * last_cross_track_error_;
            last_v_goal_y_ = k_goal_y_ * (goal_y_ - y);

            const double attraction_scale =
                rotational_apf_active_
                    ? 0.15
                    : (danger_active ? 0.35 : 1.0);

            v_att_y =
                attraction_scale *
                (last_v_path_y_ + last_v_goal_y_);
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
        // APF repulsion term (no heading control -> body frame == world
        // frame for this term, since the vehicle isn't being commanded to
        // yaw at all)
        // ------------------------------------------------------
        double v_rep_y = k_rep_y_ * repulsive_y_filtered_;

        // Fallback nudge: danger present but not clearly left/right (e.g.
        // something dead ahead) -> commit to the last meaningful avoidance
        // direction rather than dithering near zero.
        if (!rotational_apf_active_ &&
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

        double vx_world =
            clamp(
                vx + correction_vx_world,
                -max_forward_speed_,
                max_forward_speed_);

        double vy_world =
            clamp(
                v_att_y + correction_vy_world,
                -max_lateral_speed_,
                max_lateral_speed_);
        
        updateOscillationDetector(vy_world);
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

        const double yaw_rate_cmd = clamp(-k_yaw_recover_ * current_yaw_, -max_yaw_recover_rate_, max_yaw_recover_rate_);

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
        vpf_risk_ = cv::Mat::zeros(vpf_grid_rows_, vpf_grid_cols_, CV_32F);
        repulsive_y_ = 0.0;
        repulsive_y_filtered_ = 0.0;
        left_danger_ = 0.0;
        right_danger_ = 0.0;
        tracked_features_ = 0;

        vx_command_ = 0.0;
        vy_filtered_ = 0.0;
        last_avoid_direction_ = 1.0;
        last_vx_ = 0.0; last_vy_ = 0.0; last_vz_ = 0.0;

        rotational_apf_active_ = false;
        rotational_apf_used_ = false;
        rotational_direction_ = 0;
        rotational_activation_count_ = 0;
        rotational_last_request_time_ = 0.0;
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

        prev_vy_sign_ = 0.0;
        oscillation_count_ = 0;
        total_oscillation_count_ = 0;
        oscillation_failure_ = false;
        oscillation_active_ = false;
        oscillation_episode_start_time_ = 0.0;
        last_oscillation_time_ = 0.0;
        have_last_oscillation_time_ = false;
        have_oscillation_episode_start_ = false;

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
    double last_y_ref_;
    double last_cross_track_error_;
    double last_v_path_y_;
    double last_v_goal_y_;

    double current_yaw_;
    double current_yaw_rate_;
    double yaw_rate_filter_alpha_;

    double k_yaw_recover_;
    double max_yaw_recover_rate_;

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

    bool enable_rotational_apf_;
    bool rotational_trigger_on_trap_;
    bool rotational_apf_active_;
    bool rotational_apf_used_;
    int rotational_direction_;
    int rotational_activation_count_;

    double k_rotational_;
    double rotational_activation_risk_;
    double rotational_min_speed_;
    double rotational_max_speed_;
    double rotational_filter_alpha_;
    double rotational_release_delay_s_;
    double rotational_last_request_time_;

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