
import traceback
import os
import cv2
import numpy as np
import traceback

from detection.detectorMain import faceDetector, faceDetectorMultiFace
from alignment.alignmentMain import faceAlignment
from recognition.recoginationMain import get_embeddings


# -------------------------------------------------------------------
# LOADING SAVED MODELS
# -------------------------------------------------------------------

def getDetectorModel(config_dir):
    DETECTOR_CONFIG = os.path.join("/home/ubuntu/frvt/11/", config_dir,"detection/cfg/detector_yolov3-face.cfg") 
    DETECTOR_MODEL_WEIGHTS = os.path.join("/home/ubuntu/frvt/11/", config_dir,"detection/model-weights/detector_yolov3-wider_16000.weights")
    
    try:
        detector_net = cv2.dnn.readNetFromDarknet(DETECTOR_CONFIG, DETECTOR_MODEL_WEIGHTS)
    except Exception as e:
        print(e)
        return
    detector_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    detector_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    try:
        frame = np.zeros((416,416,3)).astype(np.uint8)
        detector_response = faceDetector(frame, detector_net)
    except Exception as e:
        print(e)
        pass
    
    return detector_net


# -------------------------------------------------------------------
# Detector Single Face
# -------------------------------------------------------------------
def detect(frame, detector_net):
    try:
        detector_response = faceDetector(frame, detector_net)
        if detector_response["result"] == 0:
            cropped_face = detector_response["data"]["cropped_face"]
            cropping_cord = detector_response["data"]["final_boxes"]
        else:
            cropped_face = [[[]]] 
            cropping_cord = []
    except Exception as e:
        print(traceback.format_exc())
        cropped_face = [[[]]] 
        cropping_cord = []
    return cropped_face, cropping_cord


# -------------------------------------------------------------------
# Alignment
# -------------------------------------------------------------------
def align(cropped_face, cropped_cord):
    try:
        alligned_face, eye_coords_wrt_original = faceAlignment(cropped_face, cropped_cord)
    except Exception as e:
        print(e)
        alligned_face = [[[]]]
        eye_coords_wrt_original = (0,0,-1,-1,-1,-1)
    return alligned_face, eye_coords_wrt_original


# -------------------------------------------------------------------
# GET EMBEDDINGS FOR SINGLE IMAGE.
# -------------------------------------------------------------------
def getEmbeddings(opencv_image, embedding_model, detector_net):
    try:
        print("CALLING DETECT")
        cropped_face, cropping_cord = detect(opencv_image, detector_net)
        print("CALLING ALIGN")
        alligned_face, eye_coords_wrt_original = align(cropped_face, cropping_cord)
        print("CALLING EMBEDDING")
        current_embedding = get_embeddings(alligned_face, embedding_model)
        print(len(current_embedding))
    except Exception as e:
        print(traceback.print_exc())
        current_embedding = [-1] * 128
        eye_coords_wrt_original = (0,0,-1,-1,-1,-1)
    return current_embedding, eye_coords_wrt_original


# -------------------------------------------------------------------
# FROM NIST -- MULTIPLE IMAGES OF SINGLE FACE
# -------------------------------------------------------------------
def create_template_multiple_images_single_face(image_tuple):
    print(type(image_tuple))
    print(len(image_tuple))
    try:
        # if len(image_tuple) == 4:
        print('inside if ')
        template_role = image_tuple[0]
        image_datas = image_tuple[1]
        detector_net = image_tuple[2]
        print('detection model type: ', type(detector_net))
        embedding_model = image_tuple[3]
        print('embedding model type: ', type(embedding_model))
        embeddings_list = []
        eyeCoords_list = []
        print('#iamges: ', len(image_datas))
        for image_data in image_datas:
            try:
                image_metadata = image_data[0]
                w = image_metadata[1]
                h = image_metadata[2]
                depth = image_metadata[3]
                channel = int(depth / 8)
                image = image_data[1]
                image = np.array(image)
                opencv_image = image.reshape((h,w,channel)).astype(np.uint8)
                embedding, eyeCoords = getEmbeddings(opencv_image, embedding_model, detector_net)
                # if not embedding[0] == -1:  #hardcode check to check if embeddings are correcct or not.
                embeddings_list.append(embedding) 
            except Exception as e:
                print(e)
                embedding = [-1] *128
                eyeCoords = (0,0,-1,-1,-1,-1)

            eyeCoords_list.append(eyeCoords)

        if not embeddings_list:
            embedding = [-1] *128
            embeddings_list.append(embedding)

        final_embeddings = np.mean(embeddings_list, axis=0).tolist()
        print('final embeddings length: ', len(final_embeddings), 'eye-coordinates length: ', len(eyeCoords_list))
        return (0, tuple(final_embeddings), tuple(eyeCoords_list))
        
        # else:
        #     embedding = [-1] *128
        #     embeddings_list = [embedding]
        #     eyeCoords = (0,0,-1,-1,-1,-1)
        #     eyeCoords_list = [eyeCoords]
        #     final_embeddings = np.mean(embeddings_list, axis=0).tolist()
        #     print('final embeddings length: ', len(final_embeddings), 'eye-coordinates length: ', len(eyeCoords_list))
        #     return (0, tuple(final_embeddings), tuple(eyeCoords_list))
    except Exception as e:
        print(e)
        embedding = [-1] *128
        embeddings_list = [embedding]
        eyeCoords = (0,0,-1,-1,-1,-1)
        eyeCoords_list = [eyeCoords]
        final_embeddings = np.mean(embeddings_list, axis=0).tolist()
        print('final embeddings length: ', len(final_embeddings), 'eye-coordinates length: ', len(eyeCoords_list))
        return (0, tuple(final_embeddings), tuple(eyeCoords_list))


# -------------------------------------------------------------------
# Detector MUltiple Face
# -------------------------------------------------------------------
def detectMultiface(frame, detector_net):
    try:
        detector_response = faceDetectorMultiFace(frame, detector_net)
        if detector_response["result"] == 0:
            cropped_faces = detector_response["data"]["cropped_faces"]
            cropping_cords = detector_response["data"]["final_boxes"]
        else:
            cropped_faces = [[[]]] 
            cropping_cords = []
    except Exception as e:
        print(traceback.format_exc())
        cropped_faces = [[[]]] 
        cropping_cords = []
    return cropped_faces, cropping_cords

# -------------------------------------------------------------------
# GET EMBEDDINGS MULTIPLE FACE SINGLE IMAGE.
# -------------------------------------------------------------------
def getEmbeddingsMultiface(opencv_image, embedding_model, detector_net):
    try:
        embeddings_list = []
        eye_coords_list = []

        cropped_faces_list, cropping_cords_list = detectMultiface(opencv_image, detector_net)
        for i,cropped_face in enumerate(cropped_faces_list):
            try:
                cropping_cord = cropping_cords_list[i]
                alligned_face, eye_coords_wrt_original = align(cropped_face, cropping_cord)
                current_embedding = tuple(get_embeddings(alligned_face, embedding_model))
            except Exception as e:
                print('getEmbeddingsMultiface 1', e)
                current_embedding = [-1] * 128
                eye_coords_wrt_original = (0,0,-1,-1,-1,-1)
            embeddings_list.append(current_embedding)
            eye_coords_list.append(eye_coords_wrt_original)
    except Exception as e:
        print('getEmbeddingsMultiface 2', e)
        print(traceback.print_exc())
        current_embedding = ([-1] * 128)
        eye_coords_wrt_original = ((0,0,-1,-1,-1,-1))
        embeddings_list.append(current_embedding)
        eye_coords_list.append(eye_coords_wrt_original)
    print('returning embedding list of length ', len(embeddings_list), ', eye-coordiante list: ', len(eye_coords_list))
    return embeddings_list, eye_coords_list



def create_template_single_images_mutiple_faces(image_tuple):
    try:
        if len(image_tuple) == 4:
            template_role = image_tuple[0]
            image_datas = image_tuple[1]
            detector_net = image_tuple[2]
            embedding_model = image_tuple[3]
            for image_data in image_datas:
                try:
                    image_metadata = image_data[0]
                    w = image_metadata[1]
                    h = image_metadata[2]
                    depth = image_metadata[3]
                    channel = int(depth / 8)
                    image = image_data[1]
                    image = np.array(image)
                    opencv_image = image.reshape((h,w,channel)).astype(np.uint8)
                    embeddings_list, eyeCoords_list = getEmbeddingsMultiface(opencv_image, embedding_model, detector_net)
                except Exception as e:
                    embedding = [-1] *128
                    embeddings_list = [embedding]
                    eyeCoords = (0,0,-1,-1,-1,-1)
                    eyeCoords_list = [eyeCoords]

            return (0, tuple(embeddings_list), tuple(eyeCoords_list))
        
        else:
            embedding = [-1] *128
            embeddings_list = [embedding]
            eyeCoords = (0,0,-1,-1,-1,-1)
            eyeCoords_list = [eyeCoords]
            return (0, tuple(embeddings_list), tuple(eyeCoords_list))
    
    except Exception as e:
        print(e)
        embedding = [-1] *128
        embeddings_list = [embedding]
        eyeCoords = (0,0,-1,-1,-1,-1)
        eyeCoords_list = [eyeCoords]
        return (0, tuple(embeddings_list), tuple(eyeCoords_list))
