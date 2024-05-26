from controlnet_aux import OpenposeDetector
from controlnet_aux.util import HWC3, resize_image
from controlnet_aux.open_pose import draw_poses
import warnings
import cv2
import numpy as np
from PIL import Image


class Detector(OpenposeDetector):
    def __call__(self, input_image, detect_resolution=512, image_resolution=512, include_body=True, include_hand=False, include_face=False, hand_and_face=None, output_type="pil", **kwargs):
        if hand_and_face is not None:
            warnings.warn("hand_and_face is deprecated. Use include_hand and include_face instead.", DeprecationWarning)
            include_hand = hand_and_face
            include_face = hand_and_face

        if "return_pil" in kwargs:
            warnings.warn("return_pil is deprecated. Use output_type instead.", DeprecationWarning)
            output_type = "pil" if kwargs["return_pil"] else "np"
        if type(output_type) is bool:
            warnings.warn("Passing `True` or `False` to `output_type` is deprecated and will raise an error in future versions")
            if output_type:
                output_type = "pil"

        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image, dtype=np.uint8)

        input_image = HWC3(input_image)
        input_image = resize_image(input_image, detect_resolution)
        H, W, C = input_image.shape
        
        poses = self.detect_poses(input_image, include_hand, include_face)
        bboxs = []
        for pose in poses:
            min_x = min(keypoint.x for keypoint in pose.body.keypoints if keypoint is not None)
            min_y = min(keypoint.y for keypoint in pose.body.keypoints if keypoint is not None)
            max_x = max(keypoint.x for keypoint in pose.body.keypoints if keypoint is not None)
            max_y = max(keypoint.y for keypoint in pose.body.keypoints if keypoint is not None)
            bboxs.append([min_x, min_y, max_x, max_y])

        canvas = draw_poses(poses, H, W, draw_body=include_body, draw_hand=include_hand, draw_face=include_face) 

        detected_map = canvas
        detected_map = HWC3(detected_map)
        
        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)

        return detected_map, bboxs
