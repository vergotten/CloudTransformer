# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-12-10 10:38:01
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2019-12-26 14:21:36
# @Email:  cshzxie@gmail.com
#
# Note:
# - Replace float -> double, kFloat -> kDouble in chamfer.cu

import os
import sys
from torch.autograd import gradcheck

import unittest
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))
from extensions.chamfer_dist import *


class TestChamferDistances(unittest.TestCase):
    def setUp(self):
        self.x = torch.rand(4, 64, 3).cuda()
        self.y = torch.rand(4, 128, 3).cuda()

    def test_ChamferDistanceL2(self):
        chamfer_l2 = ChamferDistanceL2()
        result = chamfer_l2(self.x, self.y)
        self.assertIsInstance(result, torch.Tensor)

    def test_ChamferDistanceL2_split(self):
        chamfer_l2_split = ChamferDistanceL2_split()
        result = chamfer_l2_split(self.x, self.y)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], torch.Tensor)
        self.assertIsInstance(result[1], torch.Tensor)

    def test_ChamferDistanceL1(self):
        chamfer_l1 = ChamferDistanceL1()
        result = chamfer_l1(self.x, self.y)
        self.assertIsInstance(result, torch.Tensor)


if __name__ == '__main__':
    unittest.main()

