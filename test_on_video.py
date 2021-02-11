
from torchvision import transforms
from darknet import Darknet
from utils import *

def pil_to_opencv(pil_image):
    open_cv_image = np.array(pil_image) 
    # Convert RGB to BGR 
    return open_cv_image[:, :, ::-1].copy() 


class SingleShotPoseRunner:
    def __init__(self, modelcfg, weightfile) -> None:
        model = Darknet(modelcfg)
        model.print_network()
        model.load_weights(weightfile)
        model.cuda()
        model.eval()
        self.model = model

    def draw(self, image):
        num_keypoints = self.model.num_keypoints
        num_classes     = 1
        test_width    = self.model.test_width
        test_height   = self.model.test_height
        self.shape = (test_width, test_height)
        transform=transforms.Compose([transforms.ToTensor(),])
        img = image.convert('RGB')
        if self.shape:
            img = img.resize(self.shape)
        img = transform(img)

        data = img.unsqueeze(0).cuda()
        output = self.model(data).data
        all_boxes = get_region_boxes(output, num_classes, num_keypoints)        
        all_boxes = [t.cpu() for t in all_boxes]

        corners2D_pr = np.array(np.reshape(all_boxes[:18], [-1, 2]), dtype='float32')

        corners2D_pr[:, 0] = corners2D_pr[:, 0] * 640
        corners2D_pr[:, 1] = corners2D_pr[:, 1] * 480

        print(corners2D_pr)

        opencv_image = pil_to_opencv(image)
        for xy in corners2D_pr:
            # Blue color in BGR
            color = (0, 255, 255)
  
            # Line thickness of 2 px
            thickness = 2
            opencv_image = cv2.circle(opencv_image, tuple(xy), 1, color, thickness)
        
        opencv_image = cv2.circle(opencv_image, tuple(corners2D_pr[0]), 1, (0, 0, 255), 2)

        lines = [
            # Bottom 4 corner
            (1, 2),
            (1, 3),
            (3, 4),
            (2, 4),
            
            # TOp 4 corner
            (5, 6),
            (5, 7),
            (6, 8),
            (7, 8),
            (8, 4),

            # connection
            (8, 4),
            (5, 1),
            (6, 2),
            (7, 3)
        ]
        for s, e in lines:
            opencv_image = cv2.line(opencv_image,tuple(corners2D_pr[s]), tuple(corners2D_pr[e]),(0,255,255),1)
        return opencv_image


modelcfg = "cfg/yolo-pose.cfg"
# weightfile = "backup/miku18/model.weights"
weightfile = "backup/onaho9/model.weights"

runner = SingleShotPoseRunner(modelcfg, weightfile)

import numpy as np
import cv2

cap = cv2.VideoCapture(4)

def cv2pil(imgCV):
    imgCV_RGB = imgCV[:, :, ::-1]
    imgPIL = Image.fromarray(imgCV_RGB)
    return imgPIL    

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the resulting frame
    imgPIL = cv2pil(frame)
    bdimage = runner.draw(imgPIL)

    cv2.imshow('frame',bdimage)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

