import os
from detection.utils import *


def faceDetector(frame, detector_net):
    
    # Create a 4D blob from a frame.
    # Input image to detector is 416 * 416
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT), [0, 0, 0], 1, crop=False)
            
            
    # Sets the input to the network
    detector_net.setInput(blob)

    # Runs the forward pass to get output of the output layers
    outs = detector_net.forward(get_outputs_names(detector_net))

    # Remove the bounding boxes with low confidence
    detector_data = post_process(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD)
    return detector_data


def faceDetectorMultiFace(frame, detector_net):
    
    # Create a 4D blob from a frame.
    # Input image to detector is 416 * 416
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT), [0, 0, 0], 1, crop=False)
            
            
    # Sets the input to the network
    detector_net.setInput(blob)

    # Runs the forward pass to get output of the output layers
    outs = detector_net.forward(get_outputs_names(detector_net))

    # Remove the bounding boxes with low confidence
    detector_data = post_process_multiface(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD)
    return detector_data