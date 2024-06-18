import sys
import unittest
import torch

# sys.path.append("ML/Pytorch/object_detection/metrics/")
from evaluations import non_max_suppression as nms


class TestNonMaxSuppression(unittest.TestCase):
    def setUp(self):
        # test cases we want to run
        self.t1_boxes = [
            [1, 1, 0.5, 0.45, 0.4, 0.5],
            [0.8,1, 0.5, 0.5, 0.2, 0.4],
            [0.7,1,  0.25, 0.35, 0.3, 0.1],
            [0.05,1,  0.1, 0.1, 0.1, 0.1],
        ]

        self.c1_boxes = [[1, 1, 0.5, 0.45, 0.4, 0.5], [0.7,1,  0.25, 0.35, 0.3, 0.1]]

        self.t2_boxes = [
            [1, 1, 0.5, 0.45, 0.4, 0.5],
            [0.9,2,  0.5, 0.5, 0.2, 0.4],
            [0.8,1,  0.25, 0.35, 0.3, 0.1],
            [0.05,1,  0.1, 0.1, 0.1, 0.1],
        ]

        self.c2_boxes = [
            [1, 1, 0.5, 0.45, 0.4, 0.5],
            [0.9,2,  0.5, 0.5, 0.2, 0.4],
            [0.8,1,  0.25, 0.35, 0.3, 0.1],
        ]

        self.t3_boxes = [
            [0.9,1,  0.5, 0.45, 0.4, 0.5],
            [1, 1, 0.5, 0.5, 0.2, 0.4],
            [0.8,2,  0.25, 0.35, 0.3, 0.1],
            [0.05,1,  0.1, 0.1, 0.1, 0.1],
        ]

        self.c3_boxes = [[1, 1, 0.5, 0.5, 0.2, 0.4], [0.8,2, 0.25, 0.35, 0.3, 0.1]]

        self.t4_boxes = [
            [0.9,1,  0.5, 0.45, 0.4, 0.5],
            [1, 1, 0.5, 0.5, 0.2, 0.4],
            [0.8,1,  0.25, 0.35, 0.3, 0.1],
            [0.05,1,  0.1, 0.1, 0.1, 0.1],
        ]

        self.c4_boxes = [
            [0.9,1,  0.5, 0.45, 0.4, 0.5],
            [1, 1, 0.5, 0.5, 0.2, 0.4],
            [0.8,1,  0.25, 0.35, 0.3, 0.1],
        ]

    def test_remove_on_iou(self):
        bboxes = nms(
            self.t1_boxes,
            prob_thresh=0.2,
            iou_thresh=7 / 20,
           
        )
        self.assertTrue(sorted(bboxes) == sorted(self.c1_boxes))

    def test_keep_on_class(self):
        bboxes = nms(
            self.t2_boxes,
            prob_thresh=0.2,
            iou_thresh=7 / 20,
            
        )
        # print(bboxes)
        # print("c2",self.c2_boxes)
        self.assertTrue(sorted(bboxes) == sorted(self.c2_boxes))

    def test_remove_on_iou_and_class(self):
        bboxes = nms(
            self.t3_boxes,
            prob_thresh=0.2,
            iou_thresh=7 / 20,
            
        )
        self.assertTrue(sorted(bboxes) == sorted(self.c3_boxes))

    def test_keep_on_iou(self):
        bboxes = nms(
            self.t4_boxes,
            prob_thresh=0.2,
            iou_thresh=9 / 20,
          
        )
        self.assertTrue(sorted(bboxes) == sorted(self.c4_boxes))


if __name__ == "__main__":
    print("Running Non Max Suppression Tests:")
    unittest.main()
