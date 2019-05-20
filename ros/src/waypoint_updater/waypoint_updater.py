#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32, Float32
from scipy.spatial import  KDTree
import numpy as np

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200
MAX_DECEL = 0.5
LOOKAHEAD_WPS_MASK = [0, 1, 2, 3, 4, 5, 10, 15, 20, 30, 50, 70, 110, 150, 196]#[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 20, 25, 30, 40, 50, 60, 80, 100, 120, 160, 196]

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        # TODO: Add other member variables you need below
        self.pose = None
        # self.base_waypoints = None
        self.waypoints_2d = None
        self.waypoint_tree = None
        self.base_lane = None
        self.stopline_wp_idx = -1

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        rospy.Subscriber('/obstacle_waypoint', Int32, self.obstacle_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        self.loop()

    def loop(self):
        self.rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            if self.pose and self.base_lane and self.waypoint_tree:
                #closest_waypoint_idx = self.get_closest_waypoint_idx()
                self.publish_waypoints()
            self.rate.sleep()

    def get_closest_waypoint_idx(self):
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        closest_idx = self.waypoint_tree.query([x,y], 1)[1]

        #Check if closest is ahead/behind ego
        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord = self.waypoints_2d[closest_idx - 1]

        #equation for hyperplane through closest_coords
        cl_vect = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        pos_vect = np.array([x, y])

        val = np.dot(cl_vect - prev_vect, pos_vect - cl_vect)

        if val > 0:
            closest_idx = (closest_idx + 1)% len(self.waypoints_2d)
        return closest_idx

    def publish_waypoints(self):
        """
        lane = Lane()
        lane.header = self.base_waypoints.header
        lane.waypoints = self.base_waypoints.waypoints[closest_idx:closest_idx + LOOKAHEAD_WPS]
        self.final_waypoints_pub.publish(lane)
        """
        final_lane = self.generate_lane()
        self.final_waypoints_pub.publish(final_lane)

    def generate_lane(self):
        lane = Lane()
        lane.header = self.base_lane.header

        closest_idx = self.get_closest_waypoint_idx()
        farthest_idx = min(closest_idx + LOOKAHEAD_WPS, len(self.base_lane.waypoints))
        base_waypoints = self.base_lane.waypoints[closest_idx:farthest_idx]

        if self.stopline_wp_idx == -1 or self.stopline_wp_idx >= farthest_idx or self.stopline_wp_idx < closest_idx + 1:
            lane.waypoints = self.regular_waypoints(closest_idx, farthest_idx)
        else:
            lane.waypoints = self.decelerate_waypoints(base_waypoints, closest_idx, farthest_idx)

        return lane

    def regular_waypoints(self, closest_idx, farthest_idx):
        final_waypoints = []

        for i in LOOKAHEAD_WPS_MASK:
            idx = closest_idx + i
            if idx < farthest_idx:
                wp = self.base_lane.waypoints[idx]
                final_waypoints.append(wp)

        return final_waypoints

    def decelerate_waypoints(self, waypoints, closest_idx, farthest_idx):
        temp = []
        stop_idx = max(self.stopline_wp_idx - closest_idx - 4, 0)
        dist = 0.0

        for i in LOOKAHEAD_WPS_MASK[::-1]:
            if closest_idx + i < farthest_idx:
                wp = waypoints[i]
                p = Waypoint()
                p.pose = wp.pose
                vel = 0.0

                if i < stop_idx + 1:
                    dist += self.distance(waypoints, i, stop_idx)
                    vel = math.sqrt(2 * MAX_DECEL * dist)
                    if vel < 0.3:
                        vel = 0.0

                p.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
                temp.insert(0,p)

        return temp


    def pose_cb(self, msg):
        # TODO: Implement
        self.pose = msg

    def waypoints_cb(self, waypoints):
        # TODO: Implement
        self.base_lane = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)


    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        self.stopline_wp_idx = msg.data

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        self.stopline_wp_idx = msg.data

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2 + (a.z-b.z)**2)
        dist = dl(waypoints[wp1].pose.pose.position, waypoints[wp2].pose.pose.position)
        """
        for i in range(wp1, wp2 + 1, 40):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i

        for i in range(wp1, wp2 + 1, 20):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i

        for i in range(wp1, wp2 + 1, 10):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i

        for i in range(wp1, wp2 + 1, 5):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i

        for i in range(wp1+1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        """
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
