import rosbag
import rospy
from sensor_msgs.msg import Imu
import numpy as np
import shutil
import yaml
import time
import math

NSECS_IN_SEC=int(1e9)

def imu_to_rosimu(timestamp_nsecs, omega, alpha):
    timestamp = rospy.Time(secs=timestamp_nsecs//NSECS_IN_SEC,
                           nsecs=timestamp_nsecs%NSECS_IN_SEC)

    rosimu = Imu()
    rosimu.header.stamp = timestamp
    rosimu.angular_velocity.x = omega[0]
    rosimu.angular_velocity.y = omega[1]
    rosimu.angular_velocity.z = omega[2]
    rosimu.linear_acceleration.x = alpha[0]
    rosimu.linear_acceleration.y = alpha[1]
    rosimu.linear_acceleration.z = alpha[2]

    return rosimu, timestamp

def convert_to_bag(imuPath,result_path):
    f_imu = open(imuPath)
    bag = rosbag.Bag(result_path, 'w')
    for line in f_imu.readlines():
        imuData = line.strip("\n").split(",")
        t1 = int(imuData[0])
        t2 = int(imuData[1])
        t_abs = math.fabs(t2-t1)/(1e9)
        t_avg = (t2+t1)//2
        if t_abs > 0.001:
            continue
        gyrData = [float(imuData[2]),float(imuData[3]),float(imuData[4])]
        accData = [float(imuData[5]),float(imuData[6]),float(imuData[7])]
        rosimu, timestamp = imu_to_rosimu(t_avg,gyrData,accData)
        bag.write("/imu", rosimu, timestamp)
    bag.close()

if __name__ == "__main__":
    imuPath = "/home/zby/data/oak-d/stereo_image/data/calibration/imu/imu.csv"
    resultPath = "./kalibr/imu.bag"
    convert_to_bag(imuPath,resultPath)
