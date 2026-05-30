#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from hector_uav_msgs.srv import EnableMotors


def make_pose(x, y, z, yaw_w=1.0):
    msg = PoseStamped()
    msg.header.frame_id = "world"
    msg.pose.position.x = x
    msg.pose.position.y = y
    msg.pose.position.z = z
    msg.pose.orientation.w = yaw_w
    return msg


if __name__ == "__main__":
    rospy.init_node("fly_forward_obstacle_test")

    pub = rospy.Publisher("/uav1/command/pose", PoseStamped, queue_size=10)

    rospy.loginfo("Waiting for motor service...")
    rospy.wait_for_service("/uav1/enable_motors")

    try:
        enable_motors = rospy.ServiceProxy("/uav1/enable_motors", EnableMotors)
        enable_motors(True)
        rospy.loginfo("Motors enabled.")
    except rospy.ServiceException as e:
        rospy.logerr("Could not enable motors: %s", e)

    rospy.sleep(2.0)

    rate = rospy.Rate(20)
    waypoints = [
        (2.0, -3.0, 4.0),
        (4.0, -3.0, 4.0),
        (6.0, -3.0, 4.0),
        (8.0, -3.0, 4.0),
        (10.0, -3.0, 4.0),
        (12.0, -3.0, 4.0),
    ]

    for x, y, z in waypoints:
        rospy.loginfo("Commanding UAV to x=%.2f y=%.2f z=%.2f", x, y, z)

        start = rospy.Time.now()
        while not rospy.is_shutdown() and (rospy.Time.now() - start).to_sec() < 4.0:
            pose = make_pose(x, y, z)
            pose.header.stamp = rospy.Time.now()
            pub.publish(pose)
            rate.sleep()

    rospy.loginfo("Finished forward obstacle test.")