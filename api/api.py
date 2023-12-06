from fastapi import FastAPI, File, UploadFile
from scipy.spatial import distance

import insightface
from insightface.app import FaceAnalysis


app = FastAPI()

threshold = 10

@app.post("/face-recognition")
def face_recognition(image1:UploadFile = File(...), image2:UploadFile = File(...)):

    # default model = buffalo_l
    app = FaceAnalysis(providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    print("Model loaded")


    faces1 = app.get(image1)
    faces2 = app.get(image2)

    for face1 in faces1:
        for face2 in faces2:
            dist = distance.euclidean(face1['embedding'], face2['embedding'])
            return dist
         

