import numpy as np

#cam0 to cam1
R = [[0.9999299,  -0.00135319, -0.01176308 ],
     [0.00160546,  0.99976836,  0.02146295],
     [0.01173132, -0.02148033,  0.99970044]]
T = [-0.07501164,0.00104704,0.0023254]
R = np.matrix(R)
T = np.matrix(T)
print(R)
print(T)
R_inv = np.linalg.inv(R)
T_inv = -R_inv*T.T
print(R_inv)
print(T_inv)

import os
root1_path = "/home/zby/data/oak-d/stereo_image/data/calibration_img_imu/imu/imu.csv"
root2_path = "/home/zby/data/oak-d/stereo_image/data/calibration_img_imu/imu/imu.txt"

f1 = open(root1_path)
f2 = open(root2_path,"w")
for line in f1.readlines():
     imuData = line.strip("\n").split(",")
     print(imuData)
     t1 = int(imuData[0])
     t2 = int(imuData[1])
     t_avg = str((t1+t2)//2)
     f2.write(t_avg+",")
     f2.write(imuData[2]+","+imuData[3]+","+imuData[4]+","+imuData[5]+","+imuData[6]+","+imuData[7]+"\n")
     print(t_avg)

