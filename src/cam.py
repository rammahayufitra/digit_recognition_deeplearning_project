import cv2
import torch
from torchvision import transforms
from PIL import Image, ImageDraw
from model import LeNet
import numpy as np



device = 'cpu'
model = torch.load('../public/models/model.pt', map_location=device)
checkpoint = torch.load('../public/weights/weight.pt', map_location=device)
model.load_state_dict(checkpoint)
model.to(device)
model.eval()

transform = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])


vid = cv2.VideoCapture(0)

value = []
while(True):
    ret, frame = vid.read()
    frame1 = cv2.resize(frame, (32,32), interpolation = cv2.INTER_AREA)
    image = Image.fromarray(frame).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0).to(device)
    outputs = model(image)
    label = np.array(outputs.detach()).argmax()
    if(label == 0):label = 2
    elif(label == 1):label = 3
    elif(label == 2):label = 4
    elif(label == 3):label = 5
    elif(label == 4):label = 6
    elif(label == 5):label = 7
    elif(label == 6):label = 9
    else:print("error value")    
    value.append(label)
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (00, 185)
    fontScale = 1
    color = (0, 0, 255)
    thickness = 2
    image = cv2.putText(frame, str(label), org, font, fontScale, 
                    color, thickness, cv2.LINE_AA, False)
    cv2.imshow('frame1', frame1)
    cv2.imshow('frame', frame )
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
