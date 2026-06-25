# EyeVNG Coordinate System

## Image origin
- The image origin is the top-left corner of the frame.
- Coordinates are expressed in pixel units.

## Axis directions
- x increases to the right.
- y increases downward.
- The image is treated as a 2D array with rows corresponding to y and columns corresponding to x.

## Eye coordinate conventions
- Eye positions are stored as pixel coordinates for the left and right eyes.
- The mean eye position is the average of the left and right eye coordinates.
- These coordinates reflect the estimated centre of the eye region in the image.

## Head pose conventions
- head_yaw, head_pitch, and head_roll are stored as simple frame-wise estimates derived from facial landmarks.
- These values are not yet calibrated to degrees in Version 1.
- They are intended as measurement primitives for future head impulse and VOR work.
