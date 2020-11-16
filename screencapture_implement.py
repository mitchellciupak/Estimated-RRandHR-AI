import cv2 as cv
import PIL
from PIL import ImageGrab
import numpy as np
import torch
import torchvision
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg




cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)


#sample coordinate list
coordlist = np.array([(0,0,300,300),(150,150,450,450)])

#box format method, input list of 4 floats, return square box of ints
def boxformat(inlist):
    outlist = inlist
    #unfinished
    return outlist

#predictor loop, default exit key is 'q'

while True:
    #tru, frame = cap.read()
    
    img = ImageGrab.grab(bbox= None)
    img = np.array(img)
    img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    
    v = predictor(img)
    
    coordlist = v["instances"].scores.tolist()
    length = len(coordlist)
    
    #Mask R-CNN
    
    #here
    for x0 in range(0,length,1):
        #variable calls for accuracy and detectron2 ID dictionary number
        #acc = v["instances"].scores[x0].item()
        #label = v["instances"].pred_classes[x0].item()
        
        element = v["instances"].pred_boxes[x0].tensor.tolist()[0]
        finlist = boxformat(element)
        
        personarray = img[finlist[0]:finlist[2],finlist[1]:finlist[3]]
        
        #personarray is the image from the formatted box
        
        #RR method
        
        #HR method
        
        #return data(possible index by x0, or detectron list index)
    
    
    if cv.waitKey(1) == ord('q'):
        break

cv.destroyAllWindows()



