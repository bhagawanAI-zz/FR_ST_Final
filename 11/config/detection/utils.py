import numpy as np
import cv2

# -------------------------------------------------------------------
# Parameters
# -------------------------------------------------------------------

CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
IMG_WIDTH = 416
IMG_HEIGHT = 416

# Default colors
COLOR_BLUE = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_YELLOW = (0, 255, 255)


# -------------------------------------------------------------------
# Help functions
# -------------------------------------------------------------------

# Get the names of the output layers
def get_outputs_names(net):
    # Get the names of all the layers in the network
    layers_names = net.getLayerNames()

    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layers_names[i - 1] for i in net.getUnconnectedOutLayers()]


# Draw the predicted bounding box
def draw_predict(frame, conf, left, top, right, bottom):
    # Draw a bounding box.
    cv2.rectangle(frame, (left, top), (right, bottom), COLOR_YELLOW, 2)

    text = '{:.2f}'.format(conf)

    # Display the label at the top of the bounding box
    label_size, base_line = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

    top = max(top, label_size[1])
    cv2.putText(frame, text, (left, top - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                COLOR_WHITE, 1)


def post_process(frame, outs, conf_threshold, nms_threshold):
    # Scan through all the bounding boxes output from the network and keep only
    # the ones with high confidence scores. Assign the box's class label as the
    # class with the highest score.
    
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]
    confidences = []
    boxes = []
    final_boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                width = int(detection[2] * frame_width)
                height = int(detection[3] * frame_height)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant
    # overlapping boxes with lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold,
                               nms_threshold)

    if len(indices) > 1:
        return {"result": -1, "message": "More than one Face Detected!!", "data": {}}

    for i in indices:
        i = i
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        final_boxes.append(box)
        left, top, right, bottom = refined_box(left, top, width, height)
        
        cropped_face = crop_face(frame, left, top, right, bottom)
        # draw_predict(frame, confidences[i], left, top, right, bottom)
    return {"result": 0, "message": "Done", "data": {"final_boxes":final_boxes[0], "cropped_face":cropped_face}}


def refined_box(left, top, width, height):
    right = left + width
    bottom = top + height
    left  = int(left - (left*1.02-left))
    right = int(right*1.02)
    return left, top, right, bottom


def crop_face(frame, left, top, right, bottom):    
    width = abs(right - left) 
    height = abs(bottom - top)

    final_crop_width = left+width
    final_crop_height = top+height

    image_width, image_height, _ = frame.shape

    if final_crop_width > image_width:
        final_crop_width = image_width
    if final_crop_height > image_height:
        final_crop_height = image_height
    if top < 0:
        top = 0
    if left < 0:
        left = 0

    cropped_image = frame[top:final_crop_height, left:final_crop_width].copy()
    
    return cropped_image


def post_process_multiface(frame, outs, conf_threshold, nms_threshold):
    # Scan through all the bounding boxes output from the network and keep only
    # the ones with high confidence scores. Assign the box's class label as the
    # class with the highest score.
    
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]
    confidences = []
    boxes = []
    cropped_faces = []
    final_boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                width = int(detection[2] * frame_width)
                height = int(detection[3] * frame_height)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant
    # overlapping boxes with lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold,
                               nms_threshold)

    for i in indices:
        i = i
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        final_boxes.append(box)
        left, top, right, bottom = refined_box(left, top, width, height)
        
        cropped_face = crop_face(frame, left, top, right, bottom)
        cropped_faces.append(cropped_face)

    return {"result": 0, "message": "Done", "data": {"final_boxes":final_boxes, "cropped_faces":cropped_faces}}
