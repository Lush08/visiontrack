"""
VisionTrack - ONNX Detector

Wrapper around ONNX Runtime for YOLO detection inference.
Handles model loading, session configuration, and inference.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort

from utils.postprocessing import decode_detections


class Detector:
    """ONNX Runtime YOLO detector."""

    def __init__(
        self,
        model_path: str,
        input_size: int = 640,
        num_classes: int = 80,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        class_names: Optional[list[str]] = None,
        use_cuda: bool = False,
        gpu_id: int = 0,
    ):
        """
        Args:
            model_path: Path to ONNX model
            input_size: Model input size (square)
            num_classes: Number of classes in model
            conf_threshold: Default confidence threshold
            iou_threshold: NMS IoU threshold
            class_names: Optional list of class names
            use_cuda: Whether to use CUDA execution provider
            gpu_id: GPU index
        """
        self.model_path = str(model_path)
        self.input_size = input_size
        self.num_classes = num_classes
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.class_names = class_names or []

        # Inspect model first to determine num_classes dynamically
        try:
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )

            providers = ["CPUExecutionProvider"]
            if use_cuda:
                providers = [
                    ("CUDAExecutionProvider", {"device_id": gpu_id}),
                    "CPUExecutionProvider",
                ]

            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=session_options,
                providers=providers,
            )

            # Auto-detect input size from model metadata
            input_info = self.session.get_inputs()[0]
            shape = input_info.shape
            if len(shape) == 4 and isinstance(shape[2], int):
                self.input_size = shape[2]
            self.input_name = input_info.name
            self.input_shape = shape

            # Auto-detect output
            self.output_names = [o.name for o in self.session.get_outputs()]
            self.output_shape = self.session.get_outputs()[0].shape
            print(
                f"Model loaded: {Path(self.model_path).name}\n"
                f"  Input: {input_info.name} {self.input_shape}\n"
                f"  Outputs: {self.output_names}\n"
                f"  Providers: {self.session.get_providers()}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model: {e}")

    def infer(self, tensor: np.ndarray) -> np.ndarray:
        """Run inference on a preprocessed input tensor."""
        outputs = self.session.run(self.output_names, {self.input_name: tensor})
        return outputs[0]

    def postprocess(
        self,
        output: np.ndarray,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Decode raw model output into (boxes_xyxy, scores, class_ids).

        Args:
            output: Raw ONNX output
            conf_threshold: Override confidence threshold
            iou_threshold: Override NMS IoU threshold

        Returns:
            Boxes in xyxy format (in model input space), scores, class ids
        """
        conf = conf_threshold if conf_threshold is not None else self.conf_threshold
        iou = iou_threshold if iou_threshold is not None else self.iou_threshold

        return decode_detections(
            output,
            num_classes=self.num_classes,
            conf_threshold=conf,
            iou_threshold=iou,
            input_size=self.input_size,
        )