import streamlit as st
import os
import glob
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from utils import U2NET, normPRED  # Import your U2NET model class

from PIL import Image
from skimage import io


st.title("Background Removal App")

# Load the U2NET model (modify the path accordingly)
model_dir = "./object_amplifier/app/u2net.pth"
net = U2NET(3, 1)  # Modify input and output channels according to your model
net.load_state_dict(torch.load(model_dir, map_location=torch.device('cpu')))
net.eval()

# Function to remove background
def remove_background(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([transforms.Resize((320, 320)),
                                    transforms.ToTensor()])
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        prediction = net(image_tensor)
        prediction = normPRED(prediction)

    return prediction

# Upload image through Streamlit
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_image:
    st.image(uploaded_image, caption="Original Image", use_column_width=True)

    if st.button("Remove Background"):
        st.text("Processing...")

        # Process the uploaded image to remove the background
        predict = remove_background(uploaded_image)
        predict = predict.squeeze()
        predict_np = predict.cpu().data.numpy()

        im = Image.fromarray(predict_np*255).convert('RGB')
        image = io.imread(uploaded_image)
        imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)
        imo.save('file_no_background.png')

        # Ensure the images are in "RGBA" mode
        subimage = Image.open('file_no_background.png').convert("RGBA")
        original = Image.open(uploaded_image).convert("RGBA")

        subdata=subimage.getdata()
        ogdata=original.getdata()

        newdata=[]
        for i in range(subdata.size[0]*subdata.size[1]):
            if subdata[i][0]==0 and subdata[i][1]==0 and subdata[i][2]==0:
                newdata.append((255,255,255,0))
            else:
                newdata.append(ogdata[i])
        subimage.putdata(newdata)
        subimage.save("file.png","PNG")   

        images=Image.open("file.png")

        st.image(images, caption="Final Image", use_column_width=True)

        os.remove("file.png")
        os.remove("file_no_background.png")

        