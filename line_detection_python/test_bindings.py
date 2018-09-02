import sys
import cv2
import numpy as np

sys.path.insert(0, '../../../devel/line_detection_python/lib')
import py_line_detection

sys.path.insert(0, '../python')
from tools import scenenet_utils

bgr = cv2.imread("test_images/0.jpg")
depth = cv2.imread("test_images/0.png", cv2.IMREAD_UNCHANGED)
depth = depth.astype(np.int32)

camera_model = scenenet_utils.get_camera_model()
camera_P_ = camera_model.P.astype(np.float32)

pcl = scenenet_utils.rgbd_to_pcl(bgr, depth, camera_model)
pcl = pcl.astype(np.float32)

cloud = np.empty((240, 320, 3), dtype=np.float32)
for i in range(240):
    for j in range(320):
        cloud[i, j, 0] = pcl[j + i*320, 0]
        cloud[i, j, 1] = pcl[j + i*320, 1]
        cloud[i, j, 2] = pcl[j + i*320, 2]

lines = py_line_detection.detect3DLines(bgr, depth, cloud, camera_P_)
print("Ten first 3D lines (start point, end point): \n {}".format(lines[:10]))
