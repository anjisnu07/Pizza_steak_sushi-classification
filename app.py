from flask import Flask,render_template,request
import tempfile
import os
from pathlib import Path
import torch

from torchvision import transforms,datasets
import cv2
from model_arch import TinyVGG
from PIL import Image
app=Flask(__name__)

#upload folder
upload_folder="Uploads"
os.makedirs(upload_folder,exist_ok=True)

model=torch.load("model.pth")

classes=['pizza', 'steak', 'sushi']

@app.route("/",methods=['GET'])
def home():
    return render_template('index.html')

@app.route("/landing",methods=["POST"])
def land():
    if request.method=="POST":
        if "image_file" not in request.files:
            return "No file uploaded"
        image_file = request.files['image_file']

        # For filename purpose
        if image_file.filename == '':
            return "No selected file"

        save_path=os.path.join("static",upload_folder,image_file.filename)
        image_file.save(save_path)
        
        # transforming images
        trans=transforms.Compose([
            transforms.Resize(size=(64,64)),
            transforms.ToTensor()
        ])

        # transfrming our image
        img=Image.open(save_path)
        image_tensor=trans(img)
        with torch.inference_mode():
            ans=model(image_tensor.unsqueeze(0))
        lbl=torch.argmax(torch.softmax(ans,dim=1),dim=1)
    return render_template("result.html",img_path=save_path,class_name=classes[lbl])

if(__name__=="__main__"):
    app.run(debug=True)