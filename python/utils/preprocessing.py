"""
VisionTrack - Image Preprocessing

YOLO-style preprocessing: letterbox resize, BGR→RGB, HWC→CHW, normalize.
"""

import cv2
import numpy as np


class Preprocessor:
    """YOLO-format image preprocessor for ONNX inference."""

    def __init__(self, input_size: int = 640, stride: int = 32):
        """
        Args:
            input_size: Square input size for the model (e.g., 640)
            stride: Model stride for letterbox coordinate alignment (default 32)
        """
        self.input_size = input_size
        self.stride = stride

    def letterbox(
        self,
        image: np.ndarray,
        new_shape: int = 640,
        color: tuple[int, int, int] = (114, 114, 114),
    ) -> tuple[np.ndarray, tuple[float, float], tuple[int, int]]:
        """
        Resize image with aspect ratio preserved, padding with gray bars.

        Args:
            image: Input BGR image
            new_shape: Target size (int for square, or (h, w) tuple)
            color: Padding color

        Returns:
            (padded_image, ratio, padding) where:
              ratio = (scale_x, scale_y)
              padding = (top, left) in pixels
        """
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        h, w = image.shape[:2]
        target_h, target_w = new_shape

        ratio = min(target_h / h, target_w / w)
        new_unpad_w = int(round(w * ratio))
        new_unpad_h = int(round(h * ratio))

        # Total padding per axis
        dw = target_w - new_unpad_w
        dh = target_h - new_unpad_h

        # Split padding evenly between the two sides
        left = dw // 2
        right = dw - left
        top = dh // 2
        bottom = dh - top

        if (h, w) != (new_unpad_h, new_unpad_w):
            image = cv2.resize(image, (new_unpad_w, new_unpad_h),
                               interpolation=cv2.INTER_LINEAR)

        padded = cv2.copyMakeBorder(
            image, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=color,
        )

        scale = (ratio, ratio)
        padding = (top, left)
        return padded, scale, padding

    def to_tensor(self, image: np.ndarray) -> np.ndarray:
        """
        Convert a letterboxed BGR image to a model-ready tensor.

        Steps: BGR→RGB, HWC→CHW, uint8→float32, normalize to [0,1].

        Args:
            image: Letterboxed BGR image (input_size, input_size, 3)

        Returns:
            Float32 tensor of shape (1, 3, input_size, input_size)
        """
        # Convert to RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # HWC → CHW
        chw = rgb.transpose(2, 0, 1)

        # uint8 → float32, normalize [0, 1]
        tensor = chw.astype(np.float32) / 255.0

        # Add batch dimension
        return np.expand_dims(tensor, axis=0)

    def preprocess(self, image: np.ndarray) -> tuple[np.ndarray, tuple[float, float], tuple[int, int]]:
        """
        End-to-end preprocessing: letterbox + tensor conversion.

        Args:
            image: Input BGR image

        Returns:
            (tensor, scale, padding) where tensor is model input
        """
        padded, scale, padding = self.letterbox(image, self.input_size)
        tensor = self.to_tensor(padded)
        return tensor, scale, padding

    def scale_boxes(
        self,
        boxes_xyxy: np.ndarray,
        scale: tuple[float, float],
        padding: tuple[int, int],
        original_shape: tuple[int, int],
        input_shape: tuple[int, int] = None,
    ) -> np.ndarray:
        """
        Map normalized/model-space boxes back to original image coordinates.

        Args:
            boxes_xyxy: Boxes in xyxy format within the padded input space (N, 4)
            scale: (scale_x, scale_y) from letterbox
            padding: (top, left) padding from letterbox
            original_shape: (h, w) of original image
            input_shape: (h, w) of model input (defaults to input_size)

        Returns:
            Boxes in xyxy format clipped to the original image
        """
        if input_shape is None:
            input_shape = (self.input_size, self.input_size)

        top, left = padding
        scale_y, scale_x = scale

        boxes = boxes_xyxy.copy().astype(np.float32)

        # Remove padding
        boxes[:, [0, 2]] -= left
        boxes[:, [1, 3]] -= top

        # Scale back to original
        boxes[:, [0, 2]] /= scale_x
        boxes[:, [1, 3]] /= scale_y

        # Clip to original image bounds
        oh, ow = original_shape[:2]
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, ow)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, oh)

        return boxes