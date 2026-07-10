#include <ros/ros.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/TwistStamped.h>
#include <hector_uav_msgs/EnableMotors.h>

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

//408.665 good result 80
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

        // Path-following / corridor-keeping controller.
        // This adds a global route-regulation term alongside APF avoidance.
        use_path_controller_ = true;
        k_path_y_ = 3;          // cross-track correction gain
        k_goal_y_ = 0.1;          // weaker final goal-y attraction gain
        last_y_ref_ = 0.0;
        last_cross_track_error_ = 0.0;
        last_v_path_y_ = 0.0;
        last_v_goal_y_ = 0.0;

        // Heading/yaw control settings
        use_heading_control_ = false;
        k_yaw_ = 0.8;                 // proportional yaw gain [rad/s per rad]
        max_yaw_rate_ = 0.5;          // maximum commanded yaw rate [rad/s]
        min_heading_speed_ = 0.2;     // avoid noisy heading command when almost stationary

        current_yaw_ = 0.0;
        desired_heading_ = 0.0;
        heading_error_ = 0.0;
        last_yaw_rate_cmd_ = 0.0;
        heading_control_active_ = false;
        heading_error_abs_sum_ = 0.0;
        heading_error_max_abs_ = 0.0;
        heading_sample_count_ = 0;

        goal_tolerance_ = 0.3;
        reached_goal_ = false;
        takeoff_done_ = false;

        collision_count_ = 0;
        in_collision_ = false;
        results_printed_ = false;

        // Mission result metrics
        path_length_ = 0.0;
        mission_start_time_ = 0.0;
        mission_end_time_ = 0.0;
        mission_timer_started_ = false;
        have_last_path_pose_ = false;
        last_path_x_ = 0.0;
        last_path_y_ = 0.0;
        last_path_z_ = 0.0;

        // CSV logging for post-processing and path visualisation
        path_files_opened_ = false;

        // Experiment summary logging. This CSV can be opened directly in Excel.
        results_file_path_ = "/root/catkin_ws/experiment_results.csv";

    obstacles_ = {

        // tree_8
        {-8.0,  5.0},
        {-8.0, -5.0},

        // pine_1
        {-7.0,  3.0},
        { 5.0,  3.0},
        {-2.0,  0.0},
        {10.0,  0.0},
        { 8.0,  8.0},
        {-7.0,  7.0},
        { 5.0,  7.0},
        {-2.0, 10.0},
        {10.0, 10.0},
        { 3.0,  1.0},
        {12.0, -5.0},
        {16.0,  2.0},
        { 8.0, 12.0},
        {14.0,-10.0},

        // // // arbol1
        // { 0.0,  0.0},

        // // // arbol2
        // {-2.0,  5.0},
        // {-1.0, -5.0},
        // {-5.0,  8.0},

        // additional tree_8
        {-3.0, 11.0},
        { 2.0, -9.0},
        {13.0,  5.0},
        {15.0, -2.0},
        { 6.0, 13.0},
        { 8.0,  4.0},
        {-7.0, -8.0}
    };

    obstacles_forest_test1 = {
        
        // Left entrance
        {-10.0,  6.5},
        { -8.0,  4.5},
        { -7.0,  8.5},

        // First avoidance section
        { -4.0,  7.5},
        { -3.0, 11.0},
        { -1.0, 10.0},

        // Central corridor
        {  1.0, 12.0},
        {  2.0,  8.0},
        {  4.0, 10.5},
        {  5.5,  8.0},
        {  7.0, 11.5},
        {  8.0,  9.5},

        // Near-goal section
        { 10.0, 10.0},
        { 11.5,  8.0},
        { 12.0, 11.8},
        { 14.5,  8.5},
        { 15.0, 12.0},

        // Background trees
        { -8.0, -4.5},
        {  3.0,  2.0},
        { 10.0,  0.0},
        { 15.0,  3.0}

    };

    obstacles_dense_corridor = {
    // Entrance gate
    {-12.0, 5.5},
    {-11.0, 8.0},
    {-9.5, 4.0},
    {-9.0, 9.5},

    // First dense section
    {-7.0, 6.5},
    {-6.0, 10.5},
    {-4.5, 7.5},
    {-3.5, 11.0},

    // Central slalom
    {-2.0, 8.5},
    {-0.5, 12.0},
    {1.0, 9.0},
    {2.5, 11.5},
    {4.0, 8.0},
    {5.5, 10.5},

    // Tight mid-to-goal corridor
    {7.0, 7.5},
    {8.0, 11.5},
    {9.5, 8.5},
    {10.5, 10.8},
    {12.0, 7.8},
    {12.5, 11.8},

    // Near-goal gate
    {14.5, 8.5},
    {15.0, 11.5},

    // Background / side trees
    {-8.0, -1.0},
    {-3.0, 3.0},
    {3.0, 4.0},
    {8.0, 3.0},
    {13.0, 4.5}
};

obstacles_dense_complex = {
    {-14.0, 6.5},
    {-13.0, 9.5},
    {-11.0, 5.5},
    {-10.0, 8.5},

    {-8.5, 7.5},
    {-7.0, 10.5},
    {-6.0, 6.0},
    {-5.0, 9.0},

    {-3.5, 8.0},
    {-2.5, 12.5},
    {-1.0, 6.5},
    {0.0, 10.0},
    {1.5, 7.0},
    {2.5, 11.5},

    {4.0, 8.5},
    {5.0, 12.0},
    {6.5, 6.0},
    {7.5, 10.5},
    {8.5, 7.5},

    {9.5, 11.0},
    {10.5, 8.0},
    {11.5, 12.0},
    {12.5, 9.0},
    {13.5, 10.5},
    {15.0, 11.5},

    {-12.5, 5.0},
    {-9.0, 9.5},
    {-4.0, 5.5},
    {3.0, 9.0},
    {6.5, 9.0},
    {9.0, 6.0},
    {14.0, 8.5}
};



    obstacles_ = obstacles_dense_complex;

        // Create CSV files for trajectory and obstacle plotting.
        // Output files are written in the catkin workspace root.
        openPathLoggingFiles();

        // -------------------------
        // APF mode switch
        // -------------------------
        use_rotational_apf_ = true;
        k_rot_ = 0.2;
        forward_reduction_ratio_ = 0.3;

        // Oscillation detection from commanded lateral velocity (vy) sign changes
        vy_sign_threshold_ = 0.2;
        prev_vy_sign_ = 0.0;
        oscillation_count_ = 0;
        total_oscillation_count_ = 0;
        oscillation_limit_ = 3;
        oscillation_failure_ = false;
        oscillation_failure_time_ = 30.0;

        oscillation_active_ = false;
        oscillation_memory_time_ = 2.0;
        oscillation_episode_start_time_ = 0.0;
        last_oscillation_time_ = 0.0;

        // More selective rotational APF trigger
        progress_poor_ = false;
        progress_rate_ = 0.0;
        prev_distance_to_goal_ = 0.0;
        progress_timer_start_ = 0.0;
        progress_window_ = 2.0;
        min_progress_rate_ = 0.2;

        // For display/debugging
        forward_reduction_active_ = false;
        rotational_apf_active_ = false;
        rotational_trigger_reason_ = "none";

        // -------------------------
        // Mission / flight settings
        // -------------------------
        start_x_ = -17.0;
        start_y_ = 4.0;
        goal_x_ = 20.0;
        goal_y_ = 12.0;
        goal_z_ = 1;

        // Direct velocity command settings [m/s]
        forward_speed_ = 10.0;
        vx_command_ = 0.0;
        max_forward_acceleration_ = 2.0;
        max_forward_speed_ = 25.0;
        max_lateral_speed_ = 4.0;
        max_vertical_speed_ = 0.5;

        // TTC/default avoidance behaviour
        default_avoid_y_speed_ = 1.8; //0.8
        danger_ttc_ = 25.0; //25
        last_avoid_direction_ = 1.0;

        vy_filtered_ = 0.0;
        vy_alpha_ = 0.08;

        // APF gains
        k_att_y_ = 1.8;  // 2.0
        k_rep_y_ = 2.2; //2.3
        k_z_ = 1.0;

        // Visual Potential Field settings
        vpf_grid_cols_ = 16;
        vpf_grid_rows_ = 8;
        vpf_risk_ = cv::Mat::zeros(vpf_grid_rows_, vpf_grid_cols_, CV_32F);
        vpf_ttc_weight_ = 1.0;
        vpf_flow_weight_ = 0.5;

        // ------------------------------------------------------
        // Risk-shaping mode for optical-flow/TTC risk.
        // 0 = baseline inverse TTC:        1 / (TTC + eps)
        // 1 = square-root inverse TTC:     sqrt(1 / (TTC + eps))
        // 2 = power inverse TTC:           pow(1 / (TTC + eps), alpha)
        // 3 = logistic TTC:                1 / (1 + exp(beta * (TTC - threshold)))
        //
        // Change this flag before running a batch of experiments.
        // This value is automatically saved in experiment_results.csv.
        // ------------------------------------------------------
        risk_shaping_mode_ = 2;
        inverse_ttc_epsilon_ = 0.05;
        risk_power_alpha_ = 1.5;
        logistic_beta_ = 1.5;
        logistic_threshold_ = 5.0;
        max_ttc_risk_ = 5.0;

        enable_visualization_ = false;

        // Performance monitoring
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

        // Feature detection / LK parameters
        max_corners_ = 400;
        quality_level_ = 0.005;
        min_distance_ = 5.0; //6
        block_size_ = 7;

        lk_win_size_ = cv::Size(21, 21);
        lk_max_level_ = 3;
        lk_criteria_ = cv::TermCriteria(
            cv::TermCriteria::EPS | cv::TermCriteria::COUNT,
            30,
            0.01
        );

        vel_pub_ = nh_public_.advertise<geometry_msgs::TwistStamped>("/uav1/command/twist", 10);
        if (enable_visualization_){
            vis_pub_ =
                nh_public_.advertise<sensor_msgs::Image>(
                    "/uav1/optical_flow/image", 1);
        }
        image_sub_ = nh_public_.subscribe("/uav1/front_camera/image_raw", 1,
                                          &OnlineLKAPFVelocity::imageCallback, this);
        pose_sub_ = nh_public_.subscribe("/uav1/ground_truth/state", 1,
                                         &OnlineLKAPFVelocity::poseCallback, this);


        enableMotors();

        control_timer_ = nh_public_.createTimer(ros::Duration(0.01),
                                                &OnlineLKAPFVelocity::controlLoop, this);

        ROS_INFO("Online LK + VPF + APF velocity avoidance node started.");
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

    void openPathLoggingFiles()
    {
        path_file_.open("/root/catkin_ws/uav_path.csv");
        obstacle_file_.open("/root/catkin_ws/obstacles.csv");

        if (!path_file_.is_open())
        {
            ROS_WARN("Could not open /root/catkin_ws/uav_path.csv for writing.");
            return;
        }

        if (!obstacle_file_.is_open())
        {
            ROS_WARN("Could not open /root/catkin_ws/obstacles.csv for writing.");
            path_file_.close();
            return;
        }

        // The path CSV is written during the main mission phase only,
        // i.e. after takeoff is complete.
        path_file_
            << "time,x,y,z,yaw,desired_heading,heading_error,"
            << "vx,vy,vz,yaw_rate,ttc,repulsive_y,left_danger,right_danger,"
            << "y_path_ref,cross_track_error,v_path_y,v_goal_y,"
            << "tracked_features,rotational_apf_active,progress_poor,collision_count\n";

        // Obstacle positions are static in this experiment.
        obstacle_file_ << "x,y,radius\n";
        const double collision_radius = 0.4;
        for (const auto& obs : obstacles_)
        {
            obstacle_file_ << obs.first << "," << obs.second << "," << collision_radius << "\n";
        }

        obstacle_file_.close();
        path_files_opened_ = true;

        ROS_INFO("CSV path logging enabled:");
        ROS_INFO("  /root/catkin_ws/uav_path.csv");
        ROS_INFO("  /root/catkin_ws/obstacles.csv");
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
            return static_cast<double>(image_count_ - 1) /
                   (last_image_time_ - first_image_time_);
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
        if (image_count_ > 1 && min_image_dt_ < std::numeric_limits<double>::max())
        {
            return min_image_dt_;
        }
        return 0.0;
    }

    double getMaximumImageInterval() const
    {
        if (image_count_ > 1)
        {
            return max_image_dt_;
        }
        return 0.0;
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
            case 0:
                // Baseline: inverse TTC.
                shaped_risk = inv_ttc;
                break;

            case 1:
                // Softer response: reduces aggressive jumps for low TTC.
                shaped_risk = std::sqrt(inv_ttc);
                break;

            case 2:
                // More aggressive nonlinear response when alpha > 1.
                shaped_risk = std::pow(inv_ttc, risk_power_alpha_);
                break;

            case 3:
                // Smooth threshold response. Risk rises rapidly below logistic_threshold_.
                shaped_risk =
                    1.0 / (1.0 + std::exp(logistic_beta_ * (safe_ttc - logistic_threshold_)));
                break;

            default:
                shaped_risk = inv_ttc;
                break;
        }

        // Prevent a single very small TTC estimate from dominating the whole field.
        shaped_risk = clamp(shaped_risk, 0.0, max_ttc_risk_);
        return shaped_risk;
    }

    void appendExperimentResults(const std::string& termination_reason,
                                 double final_x,
                                 double final_y,
                                 double final_z)
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
            ROS_WARN("Could not open %s for appending experiment results.",
                     results_file_path_.c_str());
            return;
        }

        if (write_header)
        {
            results_file
                << "ros_time,"
                << "termination,"
                << "final_x,final_y,final_z,"
                << "path_length_m,"
                << "execution_time_s,"
                << "collisions,"
                << "tracked_features,"
                << "oscillation_count,"
                << "total_oscillations,"
                << "progress_poor,"
                << "progress_rate_mps,"
                << "path_controller_used,"
                << "final_y_path_ref,"
                << "final_cross_track_error,"
                << "k_path_y,"
                << "k_goal_y,"
                << "rotational_apf_used,"
                << "heading_control_used,"
                << "risk_shaping_function,"
                << "risk_shaping_mode,"
                << "risk_power_alpha,"
                << "logistic_beta,"
                << "logistic_threshold,"
                << "final_yaw_rad,"
                << "final_desired_heading_rad,"
                << "final_heading_error_rad,"
                << "avg_abs_heading_error_rad,"
                << "max_abs_heading_error_rad,"
                << "image_count,"
                << "avg_image_rate_hz,"
                << "min_image_interval_s,"
                << "max_image_interval_s,"
                << "avg_processing_time_ms,"
                << "max_processing_time_ms,"
                << "forward_speed_mps,"
                << "k_att_y,"
                << "k_rep_y,"
                << "danger_ttc_s,"
                << "max_lateral_speed_mps,"
                << "start_x,start_y,goal_x,goal_y,goal_z"
                << "\n";
        }

        results_file << std::fixed << std::setprecision(4);

        results_file
            << ros::Time::now().toSec() << ","
            << termination_reason << ","
            << final_x << ","
            << final_y << ","
            << final_z << ","
            << path_length_ << ","
            << getMissionExecutionTime() << ","
            << collision_count_ << ","
            << tracked_features_ << ","
            << oscillation_count_ << ","
            << total_oscillation_count_ << ","
            << boolText(progress_poor_) << ","
            << progress_rate_ << ","
            << boolText(use_path_controller_) << ","
            << last_y_ref_ << ","
            << last_cross_track_error_ << ","
            << k_path_y_ << ","
            << k_goal_y_ << ","
            << boolText(use_rotational_apf_) << ","
            << boolText(use_heading_control_) << ","
            << riskShapingName() << ","
            << risk_shaping_mode_ << ","
            << risk_power_alpha_ << ","
            << logistic_beta_ << ","
            << logistic_threshold_ << ","
            << current_yaw_ << ","
            << desired_heading_ << ","
            << heading_error_ << ","
            << getAverageHeadingError() << ","
            << heading_error_max_abs_ << ","
            << image_count_ << ","
            << getAverageImageRate() << ","
            << getMinimumImageInterval() << ","
            << getMaximumImageInterval() << ","
            << getAverageProcessingTimeMs() << ","
            << max_processing_time_ms_ << ","
            << forward_speed_ << ","
            << k_att_y_ << ","
            << k_rep_y_ << ","
            << danger_ttc_ << ","
            << max_lateral_speed_ << ","
            << start_x_ << ","
            << start_y_ << ","
            << goal_x_ << ","
            << goal_y_ << ","
            << goal_z_
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
            << current_pose_.position.x << ","
            << current_pose_.position.y << ","
            << current_pose_.position.z << ","
            << current_yaw_ << ","
            << desired_heading_ << ","
            << heading_error_ << ","
            << last_vx_ << ","
            << last_vy_ << ","
            << last_vz_ << ","
            << last_yaw_rate_cmd_ << ",";

        if (have_ttc_filtered_)
        {
            path_file_ << ttc_filtered_;
        }
        else
        {
            path_file_ << "nan";
        }

        path_file_
            << ","
            << repulsive_y_ << ","
            << left_danger_ << ","
            << right_danger_ << ","
            << last_y_ref_ << ","
            << last_cross_track_error_ << ","
            << last_v_path_y_ << ","
            << last_v_goal_y_ << ","
            << tracked_features_ << ","
            << static_cast<int>(rotational_apf_active_) << ","
            << static_cast<int>(progress_poor_) << ","
            << collision_count_ << "\n";
    }

    void poseCallback(const nav_msgs::OdometryConstPtr& msg)
    {
        current_pose_ = msg->pose.pose;
        current_yaw_ = yawFromQuaternion(current_pose_.orientation);
        have_pose_ = true;

        // Accumulate travelled path length only during the main mission phase.
        // This excludes the takeoff phase and stops once the final result condition is reached.
        if (mission_timer_started_ && !results_printed_ && !reached_goal_ && !oscillation_failure_)
        {
            const double x = current_pose_.position.x;
            const double y = current_pose_.position.y;
            const double z = current_pose_.position.z;

            if (!have_last_path_pose_)
            {
                last_path_x_ = x;
                last_path_y_ = y;
                last_path_z_ = z;
                have_last_path_pose_ = true;
            }
            else
            {
                const double dx = x - last_path_x_;
                const double dy = y - last_path_y_;
                const double dz = z - last_path_z_;

                path_length_ += std::sqrt(dx * dx + dy * dy + dz * dz);

                last_path_x_ = x;
                last_path_y_ = y;
                last_path_z_ = z;
            }
        }

        // Record the trajectory sample for later plotting.
        logPathSample();
    }

    void imageCallback(const sensor_msgs::ImageConstPtr& msg)
    {
        const auto callback_start = std::chrono::high_resolution_clock::now();

        auto finish_processing_measurement = [this, &callback_start]()
        {
            const auto callback_end = std::chrono::high_resolution_clock::now();
            const double processing_ms =
                std::chrono::duration<double, std::milli>(callback_end - callback_start).count();

            processing_time_sum_ms_ += processing_ms;
            processing_count_++;

            if (processing_ms > max_processing_time_ms_)
            {
                max_processing_time_ms_ = processing_ms;
            }
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
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        double now = msg->header.stamp.toSec();
        if (now == 0.0)
        {
            now = ros::Time::now().toSec();
        }

        // Camera-rate monitoring based on received image timestamps.
        if (first_image_time_ < 0.0)
        {
            first_image_time_ = now;
        }

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
        if (dt <= 0.0)
        {
            dt = 1.0 / 30.0;
        }

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
            if (status[i])
            {
                good_old.push_back(prev_pts_[i]);
                good_new.push_back(new_pts[i]);
            }
        }

        tracked_features_ = static_cast<int>(good_new.size());

        double ttc = 0.0;
        bool ttc_valid = estimateTTC(good_old, good_new, gray.size(), dt, ttc);

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
            if (ttc_miss_count_ > ttc_miss_limit_)
            {
                have_ttc_filtered_ = false;
            }
        }

        vpf_risk_ = computeVisualPotentialField(good_old, good_new, gray.size(), dt);
        computeVPFRepulsionY(repulsive_y_, left_danger_, right_danger_);

        if (std::abs(repulsive_y_) > 0.1)
        {
            last_avoid_direction_ = sign(repulsive_y_);
        }

        if (enable_visualization_)
        {
            cv::Mat vis = drawFlow(frame, good_old, good_new);

            sensor_msgs::ImagePtr vis_msg = cv_bridge::CvImage(msg->header, sensor_msgs::image_encodings::BGR8, vis).toImageMsg();
            vis_pub_.publish(vis_msg);
        }

        prev_gray_ = gray.clone();
        prev_time_ = now;
        cv::goodFeaturesToTrack(prev_gray_, prev_pts_, max_corners_, quality_level_, min_distance_,
                                cv::Mat(), block_size_);

        finish_processing_measurement();
    }

    cv::Mat computeVisualPotentialField(const std::vector<cv::Point2f>& old_pts,
                                        const std::vector<cv::Point2f>& new_pts,
                                        const cv::Size& image_size,
                                        double dt)
    {
        const int w = image_size.width;
        const int h = image_size.height;
        cv::Mat risk_grid = cv::Mat::zeros(vpf_grid_rows_, vpf_grid_cols_, CV_32F);

        const double cx = w / 2.0;
        const double cy = h / 2.0;

        for (size_t i = 0; i < old_pts.size(); ++i)
        {
            const double x_old = old_pts[i].x;
            const double y_old = old_pts[i].y;
            const double x_new = new_pts[i].x;
            const double y_new = new_pts[i].y;

            if (y_old < h * 0.35)
            {
                continue;
            }

            const double dx = x_new - x_old;
            const double dy = y_new - y_old;
            const double flow_mag = std::sqrt(dx * dx + dy * dy);

            if (flow_mag < 0.1)
            {
                continue;
            }

            const double rx = x_old - cx;
            const double ry = y_old - cy;
            const double r = std::sqrt(rx * rx + ry * ry);

            if (r < 20.0)
            {
                continue;
            }

            const double radial_flow = (rx * dx + ry * dy) / r;
            const double radial_velocity = radial_flow / dt;

            double ttc = ttc_max_useful_;
            if (radial_velocity > 1e-3)
            {
                ttc = r / radial_velocity;
            }

            if (ttc <= 0.1 || ttc > ttc_max_useful_)
            {
                continue;
            }

            const double ttc_risk = shapeTTCRisk(ttc);
            const double risk = vpf_flow_weight_ * flow_mag + vpf_ttc_weight_ * ttc_risk;

            int col = static_cast<int>((x_old / static_cast<double>(w)) * vpf_grid_cols_);
            int row = static_cast<int>((y_old / static_cast<double>(h)) * vpf_grid_rows_);

            col = std::max(0, std::min(vpf_grid_cols_ - 1, col));
            row = std::max(0, std::min(vpf_grid_rows_ - 1, row));

            risk_grid.at<float>(row, col) += static_cast<float>(risk);
        }

        double min_val = 0.0;
        double max_val = 0.0;
        cv::minMaxLoc(risk_grid, &min_val, &max_val);
        if (max_val > 0.0)
        {
            risk_grid /= static_cast<float>(max_val);
        }

        return risk_grid;
    }

    void computeVPFRepulsionY(double& repulsive_y, double& left_danger, double& right_danger)
    {
        left_danger = 0.0;
        right_danger = 0.0;
        repulsive_y = 0.0;

        for (int r = 0; r < vpf_risk_.rows; ++r)
        {
            for (int c = 0; c < vpf_risk_.cols; ++c)
            {
                const double risk = static_cast<double>(vpf_risk_.at<float>(r, c));
                if (risk <= 0.0)
                {
                    continue;
                }

                const double x_norm = (c + 0.5) / static_cast<double>(vpf_grid_cols_);
                const double side = x_norm - 0.5;

                if (side < 0.0)
                {
                    left_danger += risk;
                }
                else
                {
                    right_danger += risk;
                }

                repulsive_y += risk * side;
            }
        }
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

    void updateOscillationDetector(double vy)
    {
        double current_sign = 0.0;
        if (std::abs(vy) >= vy_sign_threshold_)
        {
            current_sign = sign(vy);
        }

        const double now = ros::Time::now().toSec();

        if (current_sign != 0.0)
        {
            if (prev_vy_sign_ != 0.0 && current_sign != prev_vy_sign_)
            {
                oscillation_count_++;
                total_oscillation_count_++;
                last_oscillation_time_ = now;
                have_last_oscillation_time_ = true;

                if (!have_oscillation_episode_start_)
                {
                    oscillation_episode_start_time_ = now;
                    have_oscillation_episode_start_ = true;
                }

                if (oscillation_count_ >= oscillation_limit_)
                {
                    oscillation_active_ = true;
                }

                if ((now - oscillation_episode_start_time_) > oscillation_failure_time_)
                {
                    oscillation_failure_ = true;
                }
            }
            else
            {
                oscillation_count_ = std::max(0, oscillation_count_ - 1);
            }

            prev_vy_sign_ = current_sign;
        }

        if (have_last_oscillation_time_)
        {
            if ((now - last_oscillation_time_) > oscillation_memory_time_)
            {
                oscillation_active_ = false;
                oscillation_count_ = 0;
                have_oscillation_episode_start_ = false;
            }
        }
    }

    bool estimateTTC(const std::vector<cv::Point2f>& old_pts,
                     const std::vector<cv::Point2f>& new_pts,
                     const cv::Size& image_size,
                     double dt,
                     double& ttc_out)
    {
        const int w = image_size.width;
        const int h = image_size.height;
        const double cx = w / 2.0;
        const double cy = h / 2.0;

        std::vector<double> taus;
        taus.reserve(old_pts.size());

        for (size_t i = 0; i < old_pts.size(); ++i)
        {
            const double x_old = old_pts[i].x;
            const double y_old = old_pts[i].y;
            const double x_new = new_pts[i].x;
            const double y_new = new_pts[i].y;

            if (y_old < h * 0.35)
            {
                continue;
            }

            const double rx = x_old - cx;
            const double ry = y_old - cy;
            const double flow_x = x_new - x_old;
            const double flow_y = y_new - y_old;

            const double r = std::sqrt(rx * rx + ry * ry);
            if (r < 20.0)
            {
                continue;
            }

            const double radial_flow = (rx * flow_x + ry * flow_y) / r;
            const double radial_velocity = radial_flow / dt;

            if (radial_velocity > 1e-3)
            {
                const double tau = r / radial_velocity;
                if (tau > 0.1 && tau < 20.0)
                {
                    taus.push_back(tau);
                }
            }
        }

        if (taus.size() < 5)
        {
            return false;
        }

        std::sort(taus.begin(), taus.end());
        const size_t mid = taus.size() / 2;
        if (taus.size() % 2 == 0)
        {
            ttc_out = 0.5 * (taus[mid - 1] + taus[mid]);
        }
        else
        {
            ttc_out = taus[mid];
        }

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
                            cv::Scalar(0, 255, 0),
                            2,
                            cv::LINE_8,
                            0,
                            0.3);
            cv::circle(output,
                       cv::Point(static_cast<int>(new_pts[i].x), static_cast<int>(new_pts[i].y)),
                       3,
                       cv::Scalar(0, 0, 255),
                       -1);
        }

        // Draw visual potential field as a red heatmap grid overlay.
        if (!vpf_risk_.empty())
        {
            const int h = output.rows;
            const int w = output.cols;
            const double cell_w = static_cast<double>(w) / static_cast<double>(vpf_grid_cols_);
            const double cell_h = static_cast<double>(h) / static_cast<double>(vpf_grid_rows_);

            cv::Mat overlay = output.clone();
            for (int rr = 0; rr < vpf_grid_rows_; ++rr)
            {
                for (int cc = 0; cc < vpf_grid_cols_; ++cc)
                {
                    const double risk = static_cast<double>(vpf_risk_.at<float>(rr, cc));
                    if (risk <= 0.0)
                    {
                        continue;
                    }

                    const int x1 = static_cast<int>(cc * cell_w);
                    const int y1 = static_cast<int>(rr * cell_h);
                    const int x2 = static_cast<int>((cc + 1) * cell_w);
                    const int y2 = static_cast<int>((rr + 1) * cell_h);
                    const int intensity = static_cast<int>(std::max(0.0, std::min(255.0, 255.0 * risk)));

                    cv::rectangle(overlay, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 0, intensity), -1);
                }
            }
            cv::addWeighted(overlay, 0.25, output, 0.75, 0.0, output);
        }

        std::vector<std::string> lines;
        if (!have_ttc_filtered_)
        {
            lines.push_back("TTC: not reliable");
        }
        else
        {
            lines.push_back(formatString("TTC: %.2f s", ttc_filtered_));
        }

        lines.push_back(formatString("Features: %d", tracked_features_));
        lines.push_back(formatString("Left flow: %.2f", left_danger_));
        lines.push_back(formatString("Right flow: %.2f", right_danger_));
        lines.push_back(formatString("APF rep y: %.2f", repulsive_y_));
        lines.push_back(formatString("vx: %.2f", last_vx_));
        lines.push_back(formatString("vy: %.2f", last_vy_));
        lines.push_back(formatString("vz: %.2f", last_vz_));
        lines.push_back(std::string("Heading ctrl: ") + boolText(use_heading_control_));
        lines.push_back(formatString("Yaw: %.2f rad", current_yaw_));
        lines.push_back(formatString("Desired yaw: %.2f rad", desired_heading_));
        lines.push_back(formatString("Yaw err: %.2f rad", heading_error_));
        lines.push_back(formatString("Yaw rate cmd: %.2f", last_yaw_rate_cmd_));
        lines.push_back(std::string("Reached: ") + boolText(reached_goal_));
        lines.push_back(std::string("Rot APF enabled: ") + boolText(use_rotational_apf_));
        lines.push_back(std::string("Rot APF active: ") + boolText(rotational_apf_active_));
        lines.push_back(std::string("Rot trigger: ") + rotational_trigger_reason_);
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
            cv::putText(output,
                        lines[i],
                        cv::Point(20, y0 + static_cast<int>(35 * i)),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.8,
                        cv::Scalar(255, 255, 255),
                        2);
        }

        return output;
    }

    void publishVelocity(double vx, double vy, double vz, double yaw_rate = 0.0)
    {
        geometry_msgs::TwistStamped cmd;
        cmd.header.stamp = ros::Time::now();
        cmd.header.frame_id = "world";
        cmd.twist.linear.x = vx;
        cmd.twist.linear.y = vy;
        cmd.twist.linear.z = vz;
        cmd.twist.angular.z = yaw_rate;
        vel_pub_.publish(cmd);
    }

    double desiredYOnPath(double x) const
    {
        const double dx = goal_x_ - start_x_;
        if (std::abs(dx) < 1e-6)
        {
            return goal_y_;
        }

        // Progress along the straight reference line from start to goal.
        // Clamp keeps the reference from continuing beyond the goal.
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

    void controlLoop(const ros::TimerEvent&)
    {
        if (!have_pose_)
        {
            return;
        }

        const double x = current_pose_.position.x;
        const double y = current_pose_.position.y;
        const double z = current_pose_.position.z;

        const double collision_radius = 0.4;
        bool collision_now = false;

        for (const auto& obs : obstacles_)
        {
            const double ox = obs.first;
            const double oy = obs.second;
            const double d = std::sqrt((x - ox) * (x - ox) + (y - oy) * (y - oy));
            if (d < collision_radius)
            {
                collision_now = true;
                break;
            }
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
            double vz = k_z_ * error_z;
            vz = clamp(vz, -max_vertical_speed_, max_vertical_speed_);

            last_vx_ = 0.0;
            last_vy_ = 0.0;
            last_vz_ = vz;
            publishVelocity(0.0, 0.0, vz);

            if (std::abs(error_z) < 0.2)
            {
                takeoff_done_ = true;

                // Start mission timing and path-length logging after takeoff is complete.
                mission_start_time_ = ros::Time::now().toSec();
                mission_timer_started_ = true;
                path_length_ = 0.0;
                have_last_path_pose_ = true;
                last_path_x_ = x;
                last_path_y_ = y;
                last_path_z_ = z;

                ROS_INFO("Takeoff altitude reached. Starting forward flight.");
                resetPerformanceMetrics();
            }
            return;
        }

        const double distance_to_goal = std::sqrt((goal_x_ - x) * (goal_x_ - x) +
                                                  (goal_y_ - y) * (goal_y_ - y));
        const bool danger_active = have_ttc_filtered_ && (ttc_filtered_ < danger_ttc_);

        updateProgressDetector(distance_to_goal, danger_active);

        // Since this test moves mainly in +x, stop if close to goal or if goal_x has been passed.
        const bool x_reached = x >= goal_x_ && std::abs(goal_x_ - x) < goal_tolerance_;
        const bool passed_goal_x = x >= goal_x_;
        const bool y_close = std::abs(goal_y_ - y) < 0.8;

        if (distance_to_goal < goal_tolerance_ || (passed_goal_x && y_close))
        {
            reached_goal_ = true;
        }

        if (reached_goal_ || oscillation_failure_)
        {
            if (!results_printed_)
            {
                if (mission_timer_started_ && mission_end_time_ <= 0.0)
                {
                    mission_end_time_ = ros::Time::now().toSec();
                }

                ROS_INFO("========== FINAL RESULTS ==========");
                if (reached_goal_)
                {
                    ROS_INFO("Termination reason: Goal reached");
                }
                else if (oscillation_failure_)
                {
                    ROS_INFO("Termination reason: Persistent oscillation (>30 s)");
                }

                ROS_INFO("Final Position: x=%.2f y=%.2f z=%.2f", x, y, z);
                ROS_INFO("Path Length: %.2f m", path_length_);
                ROS_INFO("Execution Time: %.2f s", getMissionExecutionTime());
                ROS_INFO("Heading Control Used: %s", boolText(use_heading_control_).c_str());
                ROS_INFO("Final Yaw: %.2f rad", current_yaw_);
                ROS_INFO("Final Desired Heading: %.2f rad", desired_heading_);
                ROS_INFO("Final Heading Error: %.2f rad", heading_error_);
                ROS_INFO("Average Absolute Heading Error: %.2f rad", getAverageHeadingError());
                ROS_INFO("Maximum Absolute Heading Error: %.2f rad", heading_error_max_abs_);
                ROS_INFO("Tracked Features: %d", tracked_features_);
                ROS_INFO("Oscillation Count: %d", oscillation_count_);
                ROS_INFO("Total Oscillation Count: %d", total_oscillation_count_);
                ROS_INFO("Progress Poor: %s", boolText(progress_poor_).c_str());
                ROS_INFO("Progress Rate: %.2f m/s", progress_rate_);
                ROS_INFO("Path Controller Used: %s", boolText(use_path_controller_).c_str());
                ROS_INFO("Final Path Reference y: %.2f m", last_y_ref_);
                ROS_INFO("Final Cross-Track Error: %.2f m", last_cross_track_error_);
                ROS_INFO("Collisions: %d", collision_count_);
                ROS_INFO("Rotational APF Used: %s", boolText(use_rotational_apf_).c_str());
                ROS_INFO("Risk Shaping Function: %s", riskShapingName().c_str());
                printPerformanceMetrics();
                ROS_INFO("===================================");
                appendExperimentResults(reached_goal_ ? "Goal reached" : "Persistent oscillation",
                                        x, y, z);
                results_printed_ = true;
                closePathLoggingFiles();
            }

            const double error_z = goal_z_ - z;
            double vz = k_z_ * error_z;
            vz = clamp(vz, -max_vertical_speed_, max_vertical_speed_);
            if (std::abs(error_z) < 0.15)
            {
                vz = 0.0;
            }

            last_vx_ = 0.0;
            last_vy_ = 0.0;
            last_vz_ = vz;
            publishVelocity(0.0, 0.0, vz);
            return;
        }

        // Forward velocity command [m/s]
        double vx = forward_speed_;

        if (distance_to_goal < 8.0)
        {
            vx = std::min(vx, 5.0);
        }

        if (distance_to_goal < 4.0)
        {
            vx = std::min(vx, 3.0);
        }

        if (distance_to_goal < 2.0)
        {
            vx = std::min(vx, 1.0);
        }

        if (distance_to_goal < 0.5)
        {
            vx = 0.3;
        }

        // Stop forward motion after passing goal_x
        if (x >= goal_x_)
        {
            vx = 0.0;
        }

        // ROS_INFO_THROTTLE(0.5,
        //     "DEBUG: have_ttc=%d, ttc=%.2f, danger_ttc=%.2f, danger_active=%d, vx_before=%.2f",
        //     have_ttc_filtered_,
        //     ttc_filtered_,
        //     danger_ttc_,
        //     danger_active,
        //     vx
        // );
        // Slow down when danger increases
        if (danger_active)
        {
            if (ttc_filtered_ < 3.0)
            {
                vx *= 0.2;
                // ROS_INFO_THROTTLE(0.5, "ttc_filtered_ < 3.0 and vx: %f", vx);
            }
            else if (ttc_filtered_ < 5.0)
            {
                vx *= 0.5;
                // ROS_INFO_THROTTLE(0.5, "ttc_filtered_ < 5.0 and vx: %f", vx);
            }
            else if (ttc_filtered_ < danger_ttc_)
            {
                vx *= 0.8;
                // ROS_INFO_THROTTLE(0.5, "ttc_filtered_ < danger_ttc_ and vx: %f", vx);
            }

        }

        vx = clamp(vx, 0.0, max_forward_speed_);

        // -----------------------------------------
        // Forward velocity rate limiter
        // -----------------------------------------
        const double dt = 0.01;
        const double max_delta = max_forward_acceleration_ * dt;

        if (vx > vx_command_)
        {
            vx_command_ = std::min(vx_command_ + max_delta, vx);
        }
        else
        {
            vx_command_ = std::max(vx_command_ - max_delta, vx);
        }
        vx = vx_command_;

        // ------------------------------------------------------
        // Path-following / corridor-keeping controller
        // ------------------------------------------------------
        // APF handles local obstacle avoidance. This additional controller
        // keeps the UAV close to the straight mission line from start to goal,
        // so after an avoidance manoeuvre it is pulled back to the route instead
        // of drifting far away in y.
        double v_att_y = 0.0;

        if (use_path_controller_)
        {
            last_y_ref_ = desiredYOnPath(x);
            last_cross_track_error_ = last_y_ref_ - y;

            last_v_path_y_ = k_path_y_ * last_cross_track_error_;
            last_v_goal_y_ = k_goal_y_ * (goal_y_ - y);

            v_att_y = last_v_path_y_ + last_v_goal_y_;
        }
        else
        {
            // Original behaviour: attract directly to final goal_y only.
            last_y_ref_ = goal_y_;
            last_cross_track_error_ = goal_y_ - y;
            last_v_path_y_ = 0.0;
            last_v_goal_y_ = k_att_y_ * (goal_y_ - y);
            v_att_y = last_v_goal_y_;
        }

        // VPF/APF lateral velocity
        double v_rep_y = k_rep_y_ * repulsive_y_;

        // Optional rotational APF tangential component
        double v_rot_y = 0.0;
        rotational_apf_active_ = false;
        forward_reduction_active_ = false;
        rotational_trigger_reason_ = "none";

        const bool forward_reduction_trigger = danger_active && (vx < forward_reduction_ratio_ * forward_speed_);
        const bool oscillation_trigger = danger_active && oscillation_active_;
        const bool progress_trigger = danger_active && progress_poor_;

        forward_reduction_active_ = forward_reduction_trigger;

        // More sophisticated rotational APF trigger:
        // Rotation is applied only when the UAV is in danger, oscillating,
        // and not making enough progress toward the goal.
        const bool rotational_trigger = use_rotational_apf_ && oscillation_trigger && progress_trigger;

        if (rotational_trigger)
        {
            const double rot_sign = last_avoid_direction_;
            v_rot_y = rot_sign * k_rot_ * std::abs(vx);
            rotational_apf_active_ = true;
            rotational_trigger_reason_ = "oscillation+poor_progress";
        }
        else if (use_rotational_apf_)
        {
            if (oscillation_trigger && !progress_trigger)
            {
                rotational_trigger_reason_ = "oscillation_no_poor_progress";
            }
            else if (progress_trigger && !oscillation_trigger)
            {
                rotational_trigger_reason_ = "poor_progress_no_oscillation";
            }
            else if (forward_reduction_trigger)
            {
                rotational_trigger_reason_ = "speed_only_not_used";
            }
            else
            {
                rotational_trigger_reason_ = "none";
            }
        }

        // If TTC says danger but obstacle is centered, keep avoiding in last chosen direction.
        if ((danger_active || (left_danger_ + right_danger_) > 0.5) &&
            std::abs(repulsive_y_) < 0.15)
        {
            v_rep_y += last_avoid_direction_ * default_avoid_y_speed_;
        }

        double vy_raw = v_att_y + v_rep_y + v_rot_y;
        vy_raw = clamp(vy_raw, -max_lateral_speed_, max_lateral_speed_);

        vy_filtered_ = vy_alpha_ * vy_raw + (1.0 - vy_alpha_) * vy_filtered_;
        const double vy = vy_filtered_;

        // ------------------------------------------------------
        // Heading-angle control
        // ------------------------------------------------------
        // The translational APF still decides vx and vy. The heading controller
        // then rotates the UAV nose toward the direction of the commanded
        // horizontal velocity vector. This is useful because the front camera is
        // body-fixed, so yawing toward the motion direction can improve what the
        // camera sees during avoidance.
        double yaw_rate_cmd = 0.0;
        const double horizontal_speed_cmd = std::sqrt(vx * vx + vy * vy);

        heading_control_active_ = false;
        if (use_heading_control_ && horizontal_speed_cmd > min_heading_speed_)
        {
            desired_heading_ = std::atan2(vy, vx);
            heading_error_ = normalizeAngle(desired_heading_ - current_yaw_);
            yaw_rate_cmd = clamp(k_yaw_ * heading_error_, -max_yaw_rate_, max_yaw_rate_);
            heading_control_active_ = true;

            const double abs_error = std::abs(heading_error_);
            heading_error_abs_sum_ += abs_error;
            heading_error_max_abs_ = std::max(heading_error_max_abs_, abs_error);
            heading_sample_count_++;
        }
        else
        {
            // When almost stationary, keep the last desired heading and command zero yaw.
            heading_error_ = normalizeAngle(desired_heading_ - current_yaw_);
            yaw_rate_cmd = 0.0;
        }

        last_yaw_rate_cmd_ = yaw_rate_cmd;

        updateOscillationDetector(vy);

        const double error_z = goal_z_ - z;
        double vz = k_z_ * error_z;
        vz = clamp(vz, -max_vertical_speed_, max_vertical_speed_);

        last_vx_ = vx;
        last_vy_ = vy;
        last_vz_ = vz;

        publishVelocity(vx, vy, vz, yaw_rate_cmd);
    }

    void printPerformanceMetrics()
    {
        double average_image_rate = 0.0;
        if (image_count_ > 1 && last_image_time_ > first_image_time_)
        {
            average_image_rate =
                static_cast<double>(image_count_ - 1) /
                (last_image_time_ - first_image_time_);
        }

        double average_processing_time_ms = 0.0;
        if (processing_count_ > 0)
        {
            average_processing_time_ms =
                processing_time_sum_ms_ / static_cast<double>(processing_count_);
        }

        ROS_INFO("Image Count: %d", image_count_);
        ROS_INFO("Average Image Rate: %.2f Hz", average_image_rate);

        if (image_count_ > 1 && min_image_dt_ < std::numeric_limits<double>::max())
        {
            ROS_INFO("Minimum Image Interval: %.3f s", min_image_dt_);
            ROS_INFO("Maximum Image Interval: %.3f s", max_image_dt_);
        }
        else
        {
            ROS_INFO("Minimum Image Interval: N/A");
            ROS_INFO("Maximum Image Interval: N/A");
        }

        ROS_INFO("Average Processing Time: %.2f ms", average_processing_time_ms);
        ROS_INFO("Maximum Processing Time: %.2f ms", max_processing_time_ms_);
    }

    void printFinalResults()
    {
        if (results_printed_)
        {
            return;
        }
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
                     current_pose_.position.x,
                     current_pose_.position.y,
                     current_pose_.position.z);
        }

        ROS_INFO("Path Length: %.2f m", path_length_);
        ROS_INFO("Execution Time: %.2f s", getMissionExecutionTime());
        ROS_INFO("Heading Control Used: %s", boolText(use_heading_control_).c_str());
        ROS_INFO("Final Yaw: %.2f rad", current_yaw_);
        ROS_INFO("Final Desired Heading: %.2f rad", desired_heading_);
        ROS_INFO("Final Heading Error: %.2f rad", heading_error_);
        ROS_INFO("Average Absolute Heading Error: %.2f rad", getAverageHeadingError());
        ROS_INFO("Maximum Absolute Heading Error: %.2f rad", heading_error_max_abs_);
        ROS_INFO("Tracked Features: %d", tracked_features_);
        ROS_INFO("Total Oscillation Count: %d", total_oscillation_count_);
        ROS_INFO("Progress Poor: %s", boolText(progress_poor_).c_str());
        ROS_INFO("Progress Rate: %.2f m/s", progress_rate_);
        ROS_INFO("Path Controller Used: %s", boolText(use_path_controller_).c_str());
        ROS_INFO("Final Path Reference y: %.2f m", last_y_ref_);
        ROS_INFO("Final Cross-Track Error: %.2f m", last_cross_track_error_);
        ROS_INFO("Collisions: %d", collision_count_);
        ROS_INFO("Rotational APF Used: %s", boolText(use_rotational_apf_).c_str());
        ROS_INFO("Risk Shaping Function: %s", riskShapingName().c_str());
        printPerformanceMetrics();
        ROS_INFO("===================================");
        if (have_pose_)
        {
            appendExperimentResults("Manual stop / node shutdown",
                                    current_pose_.position.x,
                                    current_pose_.position.y,
                                    current_pose_.position.z);
        }
        else
        {
            appendExperimentResults("Manual stop / node shutdown",
                                    0.0, 0.0, 0.0);
        }
        closePathLoggingFiles();
    }

private:
    double getMissionExecutionTime() const
    {
        if (!mission_timer_started_)
        {
            return 0.0;
        }

        const double end_time = (mission_end_time_ > 0.0)
            ? mission_end_time_
            : ros::Time::now().toSec();

        return std::max(0.0, end_time - mission_start_time_);
    }

    double getAverageHeadingError() const
    {
        if (heading_sample_count_ <= 0)
        {
            return 0.0;
        }
        return heading_error_abs_sum_ / static_cast<double>(heading_sample_count_);
    }

    static double normalizeAngle(double angle)
    {
        while (angle > M_PI)
        {
            angle -= 2.0 * M_PI;
        }
        while (angle < -M_PI)
        {
            angle += 2.0 * M_PI;
        }
        return angle;
    }

    static double yawFromQuaternion(const geometry_msgs::Quaternion& q)
    {
        // Standard yaw extraction from quaternion, using ENU/world convention.
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

    static std::string boolText(bool value)
    {
        return value ? "True" : "False";
    }

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

    bool enable_visualization_;

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

    bool use_heading_control_;
    double k_yaw_;
    double max_yaw_rate_;
    double min_heading_speed_;
    double current_yaw_;
    double desired_heading_;
    double heading_error_;
    double last_yaw_rate_cmd_;
    bool heading_control_active_;
    double heading_error_abs_sum_;
    double heading_error_max_abs_;
    int heading_sample_count_;

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
    std::vector<std::pair<double, double>> obstacles_forest_test1;
    std::vector<std::pair<double, double>> obstacles_dense_corridor;
    std::vector<std::pair<double, double>> obstacles_dense_complex;


    bool use_rotational_apf_;
    double k_rot_;
    double forward_reduction_ratio_;

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

    bool forward_reduction_active_;
    bool rotational_apf_active_;
    std::string rotational_trigger_reason_;

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
    double left_danger_;
    double right_danger_;
    int tracked_features_;
    double ttc_filtered_;
    bool have_ttc_filtered_;
    double ttc_max_useful_;
    int ttc_miss_count_;
    int ttc_miss_limit_;

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