DEBUG = False
def dprint(*args):
    if DEBUG:
        print(args)

import cv2
import numpy as np
import mediapipe as mp
import itertools

from alignment.utils import normalized_to_pixel_coordinates


def AffignTransform(LEFT_NORM_COORDS_LIST, RIGHT_NORM_COORDS_LIST, cropped_face, cropped_cord):
    # compute the desired right eye x-coordinate based on the
    # desired x-coordinate of the left eye
    #  --- PARAMETERS --- 
    desiredLeftEye=(0.3, 0.3)
    desiredFaceWidth=224
    desiredFaceHeight=224
    desiredRightEyeX = 1.0 - desiredLeftEye[0]

    left, top, *_ = cropped_cord

    left_eye_center_x = int(sum([c[0] for c in LEFT_NORM_COORDS_LIST]) / len(LEFT_NORM_COORDS_LIST))
    left_eye_center_y = int(sum([c[1] for c in LEFT_NORM_COORDS_LIST]) / len(LEFT_NORM_COORDS_LIST))

    right_eye_center_x = int(sum([c[0] for c in RIGHT_NORM_COORDS_LIST]) / len(RIGHT_NORM_COORDS_LIST))
    right_eye_center_y = int(sum([c[1] for c in RIGHT_NORM_COORDS_LIST]) / len(RIGHT_NORM_COORDS_LIST))
    
    org_left = (left_eye_center_x + left, left_eye_center_y+top)
    org_right =  (right_eye_center_x + left, right_eye_center_y+top)

    # compute the angle between the eye centroids
    dY = right_eye_center_y - left_eye_center_y
    dX = right_eye_center_x - left_eye_center_x
    angle = np.degrees(np.arctan2(dY, dX)) - 180

    # determine the scale of the new resulting image by taking
    # the ratio of the distance between eyes in the *current*
    # image to the ratio of distance between eyes in the
    # *desired* image
    dist = np.sqrt((dX ** 2) + (dY ** 2))
    desiredDist = (desiredRightEyeX - desiredLeftEye[0])
    desiredDist *= desiredFaceWidth
    scale = desiredDist / dist

    # compute center (x, y)-coordinates (i.e., the median point)
    # between the two eyes in the input image
    eyesCenter = ((left_eye_center_x + right_eye_center_x) // 2, (left_eye_center_y + right_eye_center_y) // 2)
    # grab the rotation matrix for rotating and scaling the face
    M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)
    # update the translation component of the matrix
    tX = desiredFaceWidth * 0.5
    tY = desiredFaceHeight * desiredLeftEye[1]
    M[0, 2] += (tX - eyesCenter[0])
    M[1, 2] += (tY - eyesCenter[1])

    # apply the affine transformation
    (w, h) = (desiredFaceWidth, desiredFaceHeight)
    output = cv2.warpAffine(cropped_face, M, (w, h), flags=cv2.INTER_CUBIC)

    return output, (1,1,org_left[0],org_left[1],org_right[0], org_right[1]) # here  1,1 means found both eyes


def faceAlignment(cropped_face, cropped_cord):
    dprint("inside faceAlignment function, cropped_face's shape: ", cropped_face.shape)
    mp_face_mesh = mp.solutions.face_mesh
    dprint("mp_face_mesh created !")
    face_mesh_images = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)
    dprint("face_mesh_images: ", face_mesh_images)
    LEFT_EYE_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_LEFT_EYE)))
    dprint("LEFT_EYE_INDEXES: ", LEFT_EYE_INDEXES)
    RIGHT_EYE_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_RIGHT_EYE)))
    dprint("LEFT_EYE_INDEXES: ", LEFT_EYE_INDEXES)
    
    image_height, image_width, _ = cropped_face.shape
    dprint("image height, width: ", image_height, image_width)
    dprint('face_mech_images:: ', face_mesh_images)
    face_mesh_results = face_mesh_images.process(cropped_face[:,:,::-1]) # passing rgb image
    dprint("face_mesh_results: ", face_mesh_results)
    LEFT_NORM_COORDS_LIST = []
    RIGHT_NORM_COORDS_LIST = [] 
    if face_mesh_results.multi_face_landmarks[0]:
        dprint("type(face_mesh_results.multi_face_landmarks[0]): ", type(face_mesh_results.multi_face_landmarks[0]))
        face_landmarks = face_mesh_results.multi_face_landmarks[0]
        dprint("left eye loop")
        for LEFT_EYE_INDEX in LEFT_EYE_INDEXES:
            normalized_x = face_landmarks.landmark[LEFT_EYE_INDEX].x
            normalized_y = face_landmarks.landmark[LEFT_EYE_INDEX].y
            norm_xy = normalized_to_pixel_coordinates(normalized_x, normalized_y, image_width, image_height)
            LEFT_NORM_COORDS_LIST.append(norm_xy)
        dprint("right eye loop")
        for RIGHT_EYE_INDEXE in RIGHT_EYE_INDEXES:
            normalized_x = face_landmarks.landmark[RIGHT_EYE_INDEXE].x
            normalized_y = face_landmarks.landmark[RIGHT_EYE_INDEXE].y
            norm_xy = normalized_to_pixel_coordinates(normalized_x, normalized_y, image_width, image_height)
            RIGHT_NORM_COORDS_LIST.append(norm_xy)
    
    dprint("returning ...")
    return AffignTransform(LEFT_NORM_COORDS_LIST, RIGHT_NORM_COORDS_LIST, cropped_face, cropped_cord)