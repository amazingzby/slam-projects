import rosbag
import rospy
from cv_bridge import CvBridge

import glob
import cv2
import os

NSECS_IN_SEC=int(1e9)
bridge = CvBridge()

def img_to_rosimg(img, timestamp_nsecs, compress = True, resize = []):
    timestamp = rospy.Time(secs=timestamp_nsecs//NSECS_IN_SEC,
                           nsecs=timestamp_nsecs%NSECS_IN_SEC)

    if len(img.shape) == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = img

    if resize:
        gray_img = cv2.resize(gray_img, tuple(resize), cv2.INTER_AREA)
        assert gray_img.shape[0] == resize[1]

    if compress:
        rosimage = bridge.cv2_to_compressed_imgmsg(gray_img, dst_format='png')
    else:
        rosimage = bridge.cv2_to_imgmsg(gray_img, encoding="mono8")
    rosimage.header.stamp = timestamp

    return rosimage, timestamp, (gray_img.shape[1], gray_img.shape[0])

def convert_to_bag(img_dir,result_path,subsample=1, compress_img=False, compress_bag=False, resize = []):
    topic1 = "/camera/image_1"
    topic2 = "/camera/image_2"
    bag = rosbag.Bag(result_path, 'w', compression='lz4' if compress_bag else 'none')
    imgList1 = list(glob.glob(os.path.join(img_dir,"left/*.png")))
    imgList2 = list(glob.glob(os.path.join(img_dir,"right/*.png")))
    print("total images:left %d,right %d"%(len(imgList1),len(imgList2)))
    for idx1 in range(len(imgList1)):
        if(idx1 % subsample) == 0:
            time_ns = int(imgList1[idx1].split("/")[-1][:-4])
            img1 = cv2.imread(imgList1[idx1],0)
            rosimg, timestamp, resolution = img_to_rosimg(img1,time_ns,compress=compress_img,resize=resize)
            bag.write(topic1, rosimg, timestamp)
    for idx2 in range(len(imgList2)):
        if(idx2 % subsample) == 0:
            time_ns = int(imgList2[idx2].split("/")[-1][:-4])
            img2 = cv2.imread(imgList2[idx2], 0)
            rosimg, timestamp, resolution = img_to_rosimg(img2, time_ns, compress=compress_img, resize=resize)
            bag.write(topic2, rosimg, timestamp)
    bag.close()
    return resize



if __name__ == "__main__":
    root_dir = "/home/zby/data/oak-d/stereo_image/data/calibration"
    result_path = "oak_d.bag"
    convert_to_bag(root_dir, result_path, subsample=10)