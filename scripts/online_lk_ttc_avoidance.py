#!/usr/bin/env python

import rospy
import cv2
import numpy as np

from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped 
from cv_bridge import CvBridge
from hector_uav_msgs.srv import EnableMotors


class OnlineLKAPFVelocity:
    def __init__(self):
        rospy.init_node("online_lk_apf_velocity_avoidance")

        self.bridge = CvBridge()
        self.last_vx = 0.0
        self.last_vy = 0.0
        self.last_vz = 0.0

        self.prev_gray = None
        self.prev_pts = None
        self.prev_time = None
        self.current_pose = None

        self.goal_tolerance = 0.5
        self.reached_goal = False

        # -------------------------
        # Mission / flight settings
        # -------------------------
        self.start_x = -10.0
        self.start_y = -0.0
        self.goal_x = 10.0
        self.goal_y = 4.0
        self.goal_z = 1.5

        # Direct velocity command settings [m/s]
        self.forward_speed = 2.0
        self.max_forward_speed = 25.0
        self.max_lateral_speed = 3 
        self.max_vertical_speed = 2.0

        # TTC/default avoidance behaviour
        self.default_avoid_y_speed = 0.8
        self.danger_ttc = 10.0
        self.last_avoid_direction = 1.0

        self.vy_filtered = 0.0
        self.vy_alpha = 0.15
        
        # -------------------------
        # APF gains
        # -------------------------
        self.k_att_y = 3       # pulls UAV back to goal_y corridor as lateral velocity
        self.k_rep_y = 0.1##     # turns optical-flow imbalance into lateral velocity
        self.k_z = 1.0           # altitude hold gain

        # Current APF state computed from optical flow
        self.repulsive_y = 0.0
        self.left_danger = 0.0
        self.right_danger = 0.0
        self.tracked_features = 0
        self.ttc_filtered = None

        self.ttc_max_useful = 10.0
        self.ttc_miss_count = 0
        self.ttc_miss_limit = 5

        # -------------------------
        # Feature detection / LK parameters
        # -------------------------
        self.feature_params = dict(
            maxCorners=200,
            qualityLevel=0.01,
            minDistance=7,
            blockSize=7
        )

        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        self.vel_pub = rospy.Publisher(
            "/uav1/command/twist",
            TwistStamped,
            queue_size=10
        )

        self.vis_pub = rospy.Publisher(
            "/uav1/optical_flow/image",
            Image,
            queue_size=1
        )

        rospy.Subscriber(
            "/uav1/front_camera/image_raw",
            Image,
            self.image_callback,
            queue_size=1
        )

        rospy.Subscriber(
            "/uav1/ground_truth/state",
            Odometry,
            self.pose_callback,
            queue_size=1
        )

        self.enable_motors()

        # This is the velocity command rate. It no longer determines forward speed.
        # Forward speed is now directly self.forward_speed in m/s.
        rospy.Timer(rospy.Duration(0.01), self.control_loop)  # 100 Hz

        rospy.loginfo("Online LK + APF velocity avoidance node started.")

    def enable_motors(self):
        rospy.loginfo("Waiting for motor service...")
        try:
            rospy.wait_for_service("/uav1/enable_motors", timeout=10.0)
            enable_motors = rospy.ServiceProxy("/uav1/enable_motors", EnableMotors)
            enable_motors(True)
            rospy.loginfo("Motors enabled.")
        except Exception as e:
            rospy.logwarn("Could not enable motors automatically: %s", str(e))

    def pose_callback(self, msg):
        self.current_pose = msg.pose.pose

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        now = msg.header.stamp.to_sec()
        if now == 0:
            now = rospy.Time.now().to_sec()

        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_time = now
            self.prev_pts = cv2.goodFeaturesToTrack(
                self.prev_gray,
                mask=None,
                **self.feature_params
            )
            return

        dt = now - self.prev_time
        if dt <= 0:
            dt = 1.0 / 60.0 #check with 60!!!!!!!!!!

        if self.prev_pts is None or len(self.prev_pts) < 50:
            self.prev_pts = cv2.goodFeaturesToTrack(
                self.prev_gray,
                mask=None,
                **self.feature_params
            )

        if self.prev_pts is None:
            self.prev_gray = gray
            self.prev_time = now
            return

        new_pts, status, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            gray,
            self.prev_pts,
            None,
            **self.lk_params
        )

        if new_pts is None or status is None:
            self.prev_gray = gray
            self.prev_time = now
            self.prev_pts = None
            return

        good_old = self.prev_pts[status.flatten() == 1]
        good_new = new_pts[status.flatten() == 1]
        self.tracked_features = len(good_new)

        ttc = self.estimate_ttc(good_old, good_new, gray.shape, dt)

        if ttc is not None and ttc < self.ttc_max_useful:
            self.ttc_miss_count = 0

            if self.ttc_filtered is None:
                self.ttc_filtered = ttc
            else:
                alpha = 0.2
                self.ttc_filtered = alpha * ttc + (1.0 - alpha) * self.ttc_filtered

        else:
            self.ttc_miss_count += 1

            if self.ttc_miss_count > self.ttc_miss_limit:
                self.ttc_filtered = None

        self.repulsive_y, self.left_danger, self.right_danger = self.compute_flow_repulsion(
            good_old,
            good_new,
            gray.shape
        )

        if abs(self.repulsive_y) > 0.1:
            self.last_avoid_direction = np.sign(self.repulsive_y)

        vis = self.draw_flow(frame, good_old, good_new, self.ttc_filtered)
        vis_msg = self.bridge.cv2_to_imgmsg(vis, encoding="bgr8")
        self.vis_pub.publish(vis_msg)

        self.prev_gray = gray
        self.prev_time = now
        self.prev_pts = cv2.goodFeaturesToTrack(
            self.prev_gray,
            mask=None,
            **self.feature_params
        )

    def compute_flow_repulsion(self, old_pts, new_pts, image_shape):
        h, w = image_shape
        cx = w / 2.0

        left_mags = []
        right_mags = []

        for old, new in zip(old_pts, new_pts):
            x_old, y_old = old.ravel()
            x_new, y_new = new.ravel()

            if y_old < h * 0.35:
                continue

            dx = x_new - x_old
            dy = y_new - y_old
            mag = np.sqrt(dx * dx + dy * dy)

            if mag < 0.1:
                continue

            if x_old < cx:
                left_mags.append(mag)
            else:
                right_mags.append(mag)

        left_danger = np.mean(left_mags) if len(left_mags) > 0 else 0.0
        right_danger = np.mean(right_mags) if len(right_mags) > 0 else 0.0

        # Positive means stronger danger on right side -> move positive y.
        repulsive_y = right_danger - left_danger

        return float(repulsive_y), float(left_danger), float(right_danger)

    def estimate_ttc(self, old_pts, new_pts, image_shape, dt):
        h, w = image_shape
        cx = w / 2.0
        cy = h / 2.0

        taus = []

        for old, new in zip(old_pts, new_pts):
            x_old, y_old = old.ravel()
            x_new, y_new = new.ravel()

            if y_old < h * 0.35:
                    continue
            
            rx = x_old - cx
            ry = y_old - cy

            flow_x = x_new - x_old
            flow_y = y_new - y_old

            r = np.sqrt(rx * rx + ry * ry)
            if r < 20:
                continue

            radial_flow = (rx * flow_x + ry * flow_y) / r
            radial_velocity = radial_flow / dt

            if radial_velocity > 1e-3:
                tau = r / radial_velocity
                if 0.1 < tau < 20.0:
                    taus.append(tau)

        if len(taus) < 5:
            return None

        return float(np.median(taus))

    def draw_flow(self, frame, old_pts, new_pts, ttc):
        output = frame.copy()

        for old, new in zip(old_pts, new_pts):
            x_old, y_old = old.ravel()
            x_new, y_new = new.ravel()

            cv2.arrowedLine(
                output,
                (int(x_old), int(y_old)),
                (int(x_new), int(y_new)),
                (0, 255, 0),
                2,
                tipLength=0.3
            )
            cv2.circle(output, (int(x_new), int(y_new)), 3, (0, 0, 255), -1)

        if ttc is None:
            ttc_text = "TTC: not reliable"
        else:
            ttc_text = "TTC: {:.2f} s".format(ttc)

        lines = [
            ttc_text,
            "Features: {}".format(self.tracked_features),
            "Left flow: {:.2f}".format(self.left_danger),
            "Right flow: {:.2f}".format(self.right_danger),
            "APF rep y: {:.2f}".format(self.repulsive_y),
            "vx: {:.2f}".format(self.last_vx),
            "vy: {:.2f}".format(self.last_vy),
            "vz: {:.2f}".format(self.last_vz),
            "Reached: {}".format(self.reached_goal)
        ]

        if self.current_pose is not None:
            lines.extend([
                "x: {:.2f}".format(self.current_pose.position.x),
                "y: {:.2f}".format(self.current_pose.position.y),
                "z: {:.2f}".format(self.current_pose.position.z)
            ])

        y0 = 35
        for i, text in enumerate(lines):
            cv2.putText(
                output,
                text,
                (20, y0 + 35 * i),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )



        return output

    def publish_velocity(self, vx, vy, vz):
        cmd = TwistStamped()
        cmd.header.stamp = rospy.Time.now()
        cmd.header.frame_id = "world"

        cmd.twist.linear.x = vx
        cmd.twist.linear.y = vy
        cmd.twist.linear.z = vz
        cmd.twist.angular.z = 0.0

        self.vel_pub.publish(cmd)

    def control_loop(self, event):
        if self.current_pose is None:
            return

        x = self.current_pose.position.x
        y = self.current_pose.position.y
        z = self.current_pose.position.z

        distance_to_goal = np.sqrt((self.goal_x - x) ** 2 + (self.goal_y - y) ** 2)

        danger_active = (
            self.ttc_filtered is not None
            and self.ttc_filtered < self.danger_ttc
        )

        # Since this test moves mainly in +x, stop if close to goal or if goal_x has been passed.
        x_reached = x >= self.goal_x
        y_close = abs(self.goal_y - y) < 0.8

        if distance_to_goal < self.goal_tolerance or (x_reached and y_close):
            self.reached_goal = True

        if self.reached_goal:
            self.last_vx = 0.0
            self.last_vy = 0.0
            self.last_vz = 0.0
            self.publish_velocity(0.0, 0.0, 0.0)
            return

        # Forward velocity command [m/s]
        # Base forward velocity
        if x >= self.goal_x:
            vx = 0.0
        else:
            vx = self.forward_speed

        # Slow down when danger increases
        if danger_active:

            # Very close obstacle
            if self.ttc_filtered < 2.0:
                vx *= 0.2

            # Moderate danger
            elif self.ttc_filtered < 5.0:
                vx *= 0.5

            # Early warning
            elif self.ttc_filtered < self.danger_ttc:
                vx *= 0.8

        vx = np.clip(vx, 0.0, self.max_forward_speed)

        # Attractive lateral velocity toward desired corridor/goal_y
        error_y = self.goal_y - y
        v_att_y = self.k_att_y * error_y

        # Optical-flow APF lateral velocity
        v_rep_y = self.k_rep_y * self.repulsive_y

        # If TTC says danger but obstacle is centered, keep avoiding in last chosen direction.
        if danger_active:
            if abs(self.repulsive_y) < 0.1:
                v_rep_y += self.last_avoid_direction * self.default_avoid_y_speed
        
        # Raw lateral APF velocity
        vy_raw = v_att_y + v_rep_y

        # Limit raw lateral command
        vy_raw = np.clip(vy_raw, -self.max_lateral_speed, self.max_lateral_speed)

        # Smooth lateral motion to reduce oscillations
        self.vy_filtered = (
            self.vy_alpha * vy_raw
            + (1.0 - self.vy_alpha) * self.vy_filtered
        )

        vy = self.vy_filtered

        # Simple altitude hold as vertical velocity command
        error_z = self.goal_z - z
        vz = self.k_z * error_z
        vz = np.clip(vz, -self.max_vertical_speed, self.max_vertical_speed)

        self.last_vx = vx
        self.last_vy = vy
        self.last_vz = vz

        self.publish_velocity(vx, vy, vz)


if __name__ == "__main__":
    try:
        node = OnlineLKAPFVelocity()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass