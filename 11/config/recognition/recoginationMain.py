from scipy.spatial.distance import cosine
import numpy as np
import face_recognition

def get_embeddings(alligned_face, model):
	encoding = face_recognition.face_encodings(alligned_face)
	if not encoding:
		encoding = np.zeros((1,128), dtype=np.float32)
	else:
		encoding = encoding[0]
	return encoding

 
# determine if a candidate face is a match for a known face
def is_match(known_embedding, candidate_embedding, thresh=0.5):
	# calculate distance between embeddings
	score = cosine(known_embedding, candidate_embedding)
	if score <= thresh:
		return {"isMatch": 1, "matchScore":score, "thresh":thresh}
	else:
		return {"isMatch": 0, "matchScore":score, "thresh":thresh}