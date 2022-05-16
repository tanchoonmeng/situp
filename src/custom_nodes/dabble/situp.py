"""
Custom node to show keypoints and count the number of times the person's hand is waved
"""

from tkinter import NO
from typing import Any, Dict, List, Tuple
import cv2
from peekingduck.pipeline.nodes.node import AbstractNode
import pandas as pd

# setup global constants
FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)       # opencv loads file in BGR format
YELLOW = (0, 255, 255)
THRESHOLD = 0.6               # ignore keypoints below this threshold
KP_NOSE = 0         # PoseNet's skeletal keypoints
NOSE_Y_WINDOW_SIZE = 10

def map_keypoint_to_image_coords(
   keypoint: List[float], image_size: Tuple[int, int]
) -> List[int]:
   """Second helper function to convert relative keypoint coordinates to
   absolute image coordinates.
   Keypoint coords ranges from 0 to 1
   where (0, 0) = image top-left, (1, 1) = image bottom-right.

   Args:
      bbox (List[float]): List of 2 floats x, y (relative)
      image_size (Tuple[int, int]): Width, Height of image

   Returns:
      List[int]: x, y in integer image coords
   """
   width, height = image_size[0], image_size[1]
   x, y = keypoint
   x *= width
   y *= height
   return int(x), int(y)


def draw_text(img, x, y, text_str: str, color_code):
   """Helper function to call opencv's drawing function,
   to improve code readability in node's run() method.
   """
   cv2.putText(
      img=img,
      text=text_str,
      org=(x, y),
      fontFace=cv2.FONT_HERSHEY_SIMPLEX,
      fontScale=0.4,
      color=color_code,
      thickness=2,
   )


class Node(AbstractNode):
    """Custom node to display keypoints and count number of hand waves

    Args:
        config (:obj:`Dict[str, Any]` | :obj:`None`): Node configuration.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        # setup object working variables
        self.nose_y_hist = pd.Series([], dtype=float)
        self.direction = None
        self.num_direction_changes = 0
        self.num_situps = 0

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
        """This node draws keypoints and count hand waves.

        Args:
                inputs (dict): Dictionary with keys
                   "img", "bboxes", "bbox_scores", "keypoints", "keypoint_scores".

          Returns:
                outputs (dict): Empty dictionary.
        """
       
        # get required inputs from pipeline
        img = inputs["img"]

        keypoints = inputs["keypoints"]
        keypoint_scores = inputs["keypoint_scores"]

        img_size = (img.shape[1], img.shape[0])  # image width, height

        if len(keypoints) == 0:
            return {}

        # situp detection using a simple heuristic of tracking the nose vertical movement
        the_keypoints = keypoints[0]              # image only has one person
        the_keypoint_scores = keypoint_scores[0]  # only one set of scores
        nose = None

        for i, keypoints in enumerate(the_keypoints):
            keypoint_score = the_keypoint_scores[i]

            if keypoint_score >= THRESHOLD:
                x, y = map_keypoint_to_image_coords(keypoints.tolist(), img_size)
                x_y_str = f"({x}, {y})"

                if i == KP_NOSE:
                    nose = keypoints
                    the_color = YELLOW
                else:                   # generic keypoint
                    the_color = WHITE

                draw_text(img, x, y, x_y_str, the_color)

        if nose is not None:
            if len(self.nose_y_hist) >= 2:
                print("nose", nose)
                means = self.nose_y_hist.rolling(NOSE_Y_WINDOW_SIZE).mean().tolist()
                if means[-1] < means[-2]:
                    direction = "down"
                else:
                    direction = "up"

                if self.direction is not None:
                    # check if nose changes direction
                    if direction != self.direction:
                       print("DIRECTION CHANGED!")
                       self.num_direction_changes += 1

                    # every two hand direction changes == one count
                    if self.num_direction_changes >= 2:
                       self.num_situps += 1
                       self.num_direction_changes = 0   # reset direction count

                self.direction = direction

            self.nose_y_hist = pd.concat([self.nose_y_hist, pd.Series(nose[1] )])[-(NOSE_Y_WINDOW_SIZE+1):]

            situp_str = f"#situps = {self.num_situps}"
            draw_text(img, 20, 30, situp_str, YELLOW)

        return {}