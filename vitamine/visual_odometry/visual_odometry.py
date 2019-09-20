from collections import deque
from copy import copy

from autograd import numpy as np

from vitamine.keypoints import extract_keypoints, match
from vitamine.triangulation import pose_point_from_keypoints, points_from_known_poses
from vitamine.camera_distortion import CameraModel

from vitamine.visual_odometry.pose import PoseManager, estimate_pose
from vitamine.visual_odometry.point import Points
from vitamine.visual_odometry.keyframe import Keyframes


def find_best_match(matcher, keyframes, descriptors0, active_keyframe_ids):
    max_matches = 0
    argmax_matches01 = None
    argmax_keyframe_id = None

    for keyframe_id1 in active_keyframe_ids:
        keypoints1, descriptors1 = keyframes.get_triangulated(keyframe_id1)
        matches01 = matcher(descriptors0, descriptors1)
        if len(matches01) > max_matches:
            max_matches = len(matches01)
            argmax_matches01 = matches01
            argmax_keyframe_id = keyframe_id1
    return argmax_matches01, argmax_keyframe_id


class Triangulation(object):
    def __init__(self, matcher, R0, t0, keypoints0, descriptors0):
        self.matcher = matcher
        self.R0 = R0
        self.t0 = t0
        self.keypoints0 = keypoints0
        self.descriptors0 = descriptors0

    def triangulate(self, R1, t1, keypoints1, descriptors1):
        matches01 = self.matcher(self.descriptors0, descriptors1)
        indices0, indices1 = matches01[:, 0], matches01[:, 1]

        points, valid_depth_mask = points_from_known_poses(
            self.R0, R1, self.t0, t1,
            self.keypoints0[indices0], keypoints1[indices1],
        )

        return points[valid_depth_mask], matches01[valid_depth_mask]


class Initializer(object):
    def __init__(self, matcher, keypoints0, descriptors0):
        self.matcher = matcher
        self.keypoints0 = keypoints0
        self.descriptors0 = descriptors0

    def initialize(self, keypoints1, descriptors1):
        keypoints0, descriptors0 = self.keypoints0, self.descriptors0
        matches01 = self.matcher(descriptors0, descriptors1)

        R1, t1, points, valid_depth_mask = pose_point_from_keypoints(
            keypoints0[matches01[:, 0]],
            keypoints1[matches01[:, 1]]
        )

        return R1, t1, matches01[valid_depth_mask], points[valid_depth_mask]


def match_existing(matcher, keyframes, descriptors0, keyframe_ids, matches):
    """
    Match with descriptors that already have corresponding 3D points
    """

    # 3D points have corresponding two viewpoits used for triangulation
    # To estimate the pose of the new frame, match keypoints in the new
    # frame to keypoints in the two viewpoints
    # Matched keypoints have corresponding 3D points.
    # Therefore we can estimate the pose of the new frame using the matched keypoints
    # and corresponding 3D points.
    ka, kb = keyframe_ids
    ma, mb = matches[:, 0], matches[:, 1]
    # get descriptors already matched
    _, descriptors1a = keyframes.get_keypoints(ka, ma)
    _, descriptors1b = keyframes.get_keypoints(kb, mb)
    matches01a = matcher(descriptors0, descriptors1a)
    matches01b = matcher(descriptors0, descriptors1b)

    if len(matches01a) > len(matches01b):
        return matches01a[:, 0], matches01a[:, 1]
    else:
        return matches01b[:, 0], matches01b[:, 1]


class VisualOdometry(object):
    def __init__(self, camera_parameters, distortion_model, matcher=match,
                 min_keypoints=8, min_active_keyframes=8):
        self.matcher = match
        self.min_keypoints = min_keypoints
        self.min_active_keyframes = min_active_keyframes
        self.camera_model = CameraModel(camera_parameters, distortion_model)
        self.points = Points()
        self.keyframes = Keyframes()

    def export_points(self):
        return self.points.get()

    def export_poses(self):
        return self.keyframes.get_poses()

    @property
    def reference_keyframe_id(self):
        return self.keyframes.oldest_keyframe_id

    def add(self, image):
        keypoints, descriptors = extract_keypoints(image)
        return self.try_add(keypoints, descriptors)

    def try_add(self, keypoints, descriptors):
        if len(keypoints) < self.min_keypoints:
            return False

        keypoints = self.camera_model.undistort(keypoints)

        if self.keyframes.active_size == 0:
            return self.init_keypoints(keypoints, descriptors)

        if self.keyframes.active_size == 1:
            return self.try_init_points(keypoints, descriptors)

        return self.try_add_keyframe(keypoints, descriptors)

    def try_init_points(self, keypoints1, descriptors1):
        keyframe_id0 = 0
        keypoints0, descriptors0 = self.keyframes.get_keypoints(keyframe_id0)

        init = Initializer(self.matcher, keypoints0, descriptors0)
        R1, t1, matches01, points = init.initialize(keypoints1, descriptors1)

        # if not self.inlier_condition(matches):
        #     return False
        # if not pose_condition(R1, t1, points):
        #     return False

        keyframe_id1 = self.keyframes.add(keypoints1, descriptors1, R1, t1)
        self.point_manager.add(points, (keyframe_id0, keyframe_id1), matches01)
        return True

    def init_keypoints(self, keypoints, descriptors):
        R, t = np.identity(3), np.zeros(3)
        keyframe_id = self.keyframes.add(keypoints, descriptors, R, t)

    def get_untriangulated(self, keyframe_id):
        indices = self.point_manager.get_triangulated(keyframe_id)
        return self.keyframes.get_untriangulated(keyframe_id, indices)

    def try_add_keyframe(self, keypoints0, descriptors0):
        points1, keyframe_ids, matches = self.point_manager.get(0)  # oldest
        indices0, indices1 = match_existing(
            self.matcher, self.keyframes,
            descriptors0, keyframe_ids, matches
        )
        R0, t0 = estimate_pose(points1[indices1], keypoints0[indices0])
        # if not pose_condition(R, t, points):
        #     return False

        triangulator = Triangulation(self.matcher, R0, t0,
                                     keypoints0, descriptors0)
        active_keyframe_ids = copy(self.keyframes.active_keyframe_ids)
        keyframe_id0 = self.keyframes.add(keypoints0, descriptors0, R0, t0)

        for keyframe_id1 in active_keyframe_ids:
            indices1 = self.get_untriangulated(keyframe_id1)

            if len(indices1) == 0:
                continue
            print("indices1", indices1)
            keypoints1, descriptors1 = self.keyframes.get_keypoints(
                keyframe_id1, indices1
            )

            R1, t1 = self.keyframes.get_pose(keyframe_id1)

            points, matches01 = triangulator.triangulate(R1, t1,
                                                         keypoints1, descriptors1)
            if len(matches01) == 0:
                continue

            matches01[:, 1] = indices1[matches01[:, 1]]

            self.point_manager.add(points, (keyframe_id0, keyframe_id1),
                                   matches01)
        return True

    def try_remove(self):
        if self.keyframes.n_active <= self.min_active_keyframes:
            return False

        # remove the oldest keyframe and the corresponding points
        keyframe_id = self.keyframes.oldest_keyframe_id
        self.keyframes.remove(keyframe_id)
        self.point_manager.remove(keyframe_id)
        return True
