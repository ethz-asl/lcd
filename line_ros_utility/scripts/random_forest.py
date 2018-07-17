#!/usr/bin/env python

import sklearn.ensemble as learn
import numpy as np
import rospkg
import rospy
import cv2
from scipy.sparse import csr_matrix
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from line_ros_utility.srv import *

# relative path of training data from the rospackage.
path_train_data = '/../data/traj_50' # traj_51 ??

class RandomForestDistanceMeasure():
    random_forest = learn.RandomForestClassifier()
    def __init__(self):
        self.path = rospkg.RosPack().get_path('line_ros_utility')
        train_path = self.path + path_train_data
        from_txt = np.loadtxt(train_path + '/lines_with_labels_0.txt')
        for i in range(1, 299):
            temp_data = np.loadtxt(train_path + '/lines_with_labels_' + str(i) + '.txt')
            if temp_data.size is not 0:
                from_txt = np.vstack((from_txt, temp_data))
        self.data = from_txt[:, 0:-1]
        self.labels = from_txt[:,-1]
        self.random_forest.fit(self.data, self.labels)
        self.cvbridge = CvBridge()

    def return_trees(self, req):
        print 'Recieved tree request.'
        image_list = []
        for i in range(len(self.random_forest.estimators_)):
            image_list.append(self.cvbridge.cv2_to_imgmsg(
                                np.vstack((self.random_forest.estimators_[i].tree_.children_left,
                                           self.random_forest.estimators_[i].tree_.children_right)).astype(float),
                                "64FC1"))
        return TreeRequestResponse(image_list)

    def return_decision_paths(self, req):
        print 'Recieved decision path request.'
        image_list = []
        np_array = np.zeros((len(req.lines)/20, 20))
        for i in range(len(req.lines)/20):
            np_array[i, :] = req.lines[i*20:(i+1)*20]
        for i in range(len(self.random_forest.estimators_)):
            image = np.array(self.random_forest.estimators_[i].decision_path(np_array).nonzero(), dtype=float)
            image_list.append(self.cvbridge.cv2_to_imgmsg(image, "64FC1"))
        return RequestDecisionPathResponse(image_list)

def run():
    rospy.init_node('random_forest_server')
    rf = RandomForestDistanceMeasure()
    service = rospy.Service('req_trees', TreeRequest, rf.return_trees)
    service = rospy.Service('req_decision_paths', RequestDecisionPath, rf.return_decision_paths)
    rospy.spin()


if __name__ == "__main__":
    run()
