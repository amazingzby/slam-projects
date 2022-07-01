# coding: utf-8
import cv2
import numpy as np
import os
from tqdm import tqdm
import numpy

#特征提取参数
feature_params = dict(
    maxCorners=150,
    qualityLevel=0.01,
    minDistance=30,
    # blockSize=7
)
# 光流参数
lk_params = dict(
    winSize=(21, 21),
    maxLevel=3,
    # criteria=(cv.TERM_CRITERIA_EPS | cv.TermCriteria_COUNT, 10, .03),
    # minEigThreshold=1e-4
)

def oakd_sort(elem):
    elem_msg = elem.split("-")[:3]
    time_sec = (int(elem_msg[0])*60.0*60.0) + (int(elem_msg[1])*60.0) + float(elem_msg[2][:-4])
    return time_sec

def simcom_sort(elem):
    return int(elem[:-4])

#解析oak数据集内容
#输入：数据集根目录，路径下left和right文件夹分别存放左右目图片
#输出：列表，格式为N*2，每个元素为左右目分别的绝对路径
def parse_oakd_files(root_path):
    left_dir  = os.path.join(root_path,"left")
    right_dir = os.path.join(root_path,"right")
    left_files  = os.listdir(left_dir)
    right_files = os.listdir(right_dir)
    left_files.sort(key=oakd_sort)
    right_files.sort(key=oakd_sort)
    fileList = []
    for idx in range(len(left_files)):
        left_path = os.path.join(left_dir,left_files[idx])
        right_path = os.path.join(right_dir,right_files[idx])
        fileList.append([left_path,right_path])
    return fileList

def parse_simcom_files(root_path):
    left_dir  = os.path.join(root_path,"cam0/data")
    right_dir = os.path.join(root_path,"cam1/data")
    left_files = os.listdir(left_dir)
    left_files.sort(key=simcom_sort)
    fileList = []
    for idx in range(len(left_files)):
        left_path = os.path.join(left_dir,left_files[idx])
        right_path = os.path.join(right_dir,left_files[idx])
        fileList.append([left_path, right_path])
    return fileList

#鱼眼校正
def correct_fish_eye(img):
    # 创建一个空的map
    h,w = img.shape
    cameraMatrix = np.array(
        [[288.5533097647282, 0.0, 328.5774794002998], [0.0, 288.74180769013986, 251.75148410611664], [0.0, 0.0, 1.0]])
    distCoeffs = np.array([[0.3225316428493221, -1.5781071448252166, 2.6523109736433748, -1.4108433216890008]])
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(cameraMatrix, distCoeffs, np.eye(3), cameraMatrix, (w,h), cv2.CV_16SC2)
    # 进行校正
    dst = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return dst

#计算光流
#输入图片路径1，图片路径2
#输出光流追踪图片
def opt_flow(imgPath1,imgPath2,fish_eye=False):
    gray1 = cv2.imread(imgPath1,0)
    gray2 = cv2.imread(imgPath2,0)
    if fish_eye:
        gray1 = correct_fish_eye(gray1)
        gray2 = correct_fish_eye(gray2)
    p0 = cv2.goodFeaturesToTrack(gray1,mask=None, **feature_params)
    #计算光流
    p1, st, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, p0, None, **lk_params)
    good_new = p1[st == 1]
    good_old = p0[st == 1]
    gray1 = cv2.cvtColor(gray1,cv2.COLOR_GRAY2BGR)
    gray2 = cv2.cvtColor(gray2,cv2.COLOR_GRAY2BGR)
    dmatchs = []
    draw_line = True
    h, w, _ = gray1.shape
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        a, b = int(a), int(b)
        c, d = old.ravel()
        c, d = int(c), int(d)
        # mask = cv.line(mask, (a, b), (c, d),(0,255,100), 2)
        point_size = max(2,w//200)
        gray1 = cv2.circle(gray1, (c, d), point_size, (0, 100, 255), -1)
        gray2 = cv2.circle(gray2, (a, b), point_size, (0, 100, 255), -1)
    img = cv2.hconcat([gray1,gray2])
    imgMatch = img.copy()
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        a, b = int(a), int(b)
        c, d = old.ravel()
        c, d = int(c), int(d)
        rand_a = np.random.randint(0,256)
        rand_b = np.random.randint(0,256)
        rand_c = np.random.randint(0,256)

        cv2.line(imgMatch, (int(np.round(c)), int(np.round(d))), (int(np.round(a) + w),
                 int(np.round(b))), (rand_a, rand_b, rand_c),1, lineType=cv2.LINE_AA, shift=0)

    return img,imgMatch

def show_oak():
    oak_d_path = "/home/zby/data/oak-d/stereo_image/data/2022-06-24                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     "
    pair_files = parse_oakd_files(oak_d_path)
    for idx in range(len(pair_files)):
        img,imgMatch = opt_flow(pair_files[idx][0],pair_files[idx][1])
        h,w,_ = img.shape
        img = cv2.resize(img,(w//2,h//2))
        imgMatch = cv2.resize(imgMatch,(w//2,h//2))
        img_a = cv2.vconcat([img,imgMatch])
        cv2.imshow("img",img_a)
        cv2.waitKey()
        #cv2.imshow("img",imgMatch)
        #cv2.waitKey()
        cv2.destroyAllWindows()

def show_simcom():
    simcom_path = "/home/zby/data/simcom/20220225_143810/mav0"
    pair_files = parse_simcom_files(simcom_path)
    for idx in range(len(pair_files)):
        img,imgMatch = opt_flow(pair_files[idx][0],pair_files[idx][1],fish_eye=True)
        h,w,_ = img.shape
        #img = cv2.resize(img,(w//2,h//2))
        #imgMatch = cv2.resize(imgMatch, (w // 2, h // 2))
        img_a = cv2.vconcat([img,imgMatch])
        cv2.imshow("img",img_a)
        cv2.waitKey()
        #cv2.imshow("img",imgMatch)
        #cv2.waitKey()
        cv2.destroyAllWindows()

def show_euROC():
    simcom_path = "/home/zby/data/EuRoC/MH_01/mav0"
    pair_files = parse_simcom_files(simcom_path)
    for idx in range(len(pair_files)):
        img,imgMatch = opt_flow(pair_files[idx][0],pair_files[idx][1],fish_eye=False)
        h,w,_ = img.shape
        #img = cv2.resize(img,(w//2,h//2))
        #imgMatch = cv2.resize(imgMatch, (w // 2, h // 2))
        img_a = cv2.vconcat([img,imgMatch])
        cv2.imshow("img",img_a)
        cv2.waitKey()
        #cv2.imshow("img",imgMatch)
        #cv2.waitKey()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    show_oak()
    #show_simcom()
