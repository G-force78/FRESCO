import cv2
from insightface.app import FaceAnalysis
import torch

app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))


images = [""]

faceid_embeds = []
for image in images:
    image = cv2.imread("person.jpg")
    faces = app.get(image)
    faceid_embeds.append(torch.from_numpy(faces[0].normed_embedding).unsqueeze(0).unsqueeze(0))
  faceid_embeds = torch.cat(faceid_embeds, dim=1)
