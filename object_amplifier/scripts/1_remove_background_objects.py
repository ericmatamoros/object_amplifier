import os
import time
import torch
from torch.autograd import Variable

from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from loguru import logger
import glob
import os

from object_amplifier.model import U2NET, U2NETP
from object_amplifier.settings import U2NetSettings

from object_amplifier import IMAGE_PATH, MODELS_PATH
from .data_loader import (
    RescaleT,
    ToTensorLab,
    SalObjDataset
)
from ._utils import normPRED, save_output

# ------- 1. define loss function --------

def remove_background():

    ############################################################
    ########             DEFINE PARAMETERS              ########
    ############################################################
    settings = U2NetSettings()
    model_name=settings['model_name']
    image_dir=os.path.join(IMAGE_PATH, 'input')
    prediction_dir=os.path.join(IMAGE_PATH,model_name+'_results'+os.sep)
    model_dir = os.path.join(MODELS_PATH,model_name+'.pth')
    output_dir = os.path.join(IMAGE_PATH, 'output')

    img_name_list = glob.glob(image_dir + os.sep + '*')
    logger.info(img_name_list)

    ############################################################
    ########            LOAD DATA & MODEL               ########
    ############################################################
    test_salobj_dataset = SalObjDataset(img_name_list = img_name_list,
                                        lbl_name_list = [],
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)])
                                        )
                                        
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)
    
    # Define model
    if(model_name=='u2net'):
        logger.info("...load U2NET---173.6 MB")
        net = U2NET(3,1)
    elif(model_name=='u2netp'):
        logger.info("...load U2NEP---4.7 MB")
        net = U2NETP(3,1)
    
    logger.info("Loading trained model")
    net.load_state_dict(torch.load(model_dir, map_location=torch.device('cpu')))
    if torch.cuda.is_available():
        net.cuda()
    net.eval()

    ############################################################
    ########             REMOVE BACKGROUND              ########
    ############################################################
    for i_test, data_test in enumerate(test_salobj_dataloader):

        logger.info(f"Inferencing for image: {img_name_list[i_test].split('/')[-1]}")

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1,d2,d3,d4,d5,d6,d7= net(inputs_test)

        # normalization
        pred = d1[:,0,:,:]
        pred = normPRED(pred)

        # save results to test_results folder
        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir, exist_ok=True)
        save_output(img_name_list[i_test],pred,prediction_dir)

        del d1,d2,d3,d4,d5,d6,d7

    ############################################################
    ########               PREPARE OUTPUT               ########
    ############################################################
    for files in img_name_list:
        filename = files.split("/")[-1].split(".jpg")[0]
        #subimage
        subimage=Image.open(f"{prediction_dir}/{filename}.png")
        #originalimage
        original=Image.open(files)

        subimage=subimage.convert("RGBA")
        original=original.convert("RGBA")

        subdata=subimage.getdata()
        ogdata=original.getdata()

        newdata=[]
        for i in range(subdata.size[0]*subdata.size[1]):
            if subdata[i][0]==0 and subdata[i][1]==0 and subdata[i][2]==0:
                newdata.append((255,255,255,0))
            else:
                newdata.append(ogdata[i])
        subimage.putdata(newdata)
        subimage.save(f"{output_dir}/{filename}.png","PNG")
     

if __name__ == "__main__":
    remove_background()

