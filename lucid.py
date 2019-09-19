import os
from model.edsr import edsr
import matplotlib.pyplot as plt
from model import resolve_single
from PIL import Image
import numpy as np
from pathlib import Path, PureWindowsPath

# generate super resolution image
def generateSR(lr_image_path):
    lr = np.array(Image.open(lr_image_path))
    sr = resolve_single(model, lr)
    img = Image.fromarray(sr.numpy(), 'RGB')
    n = img.size
    name = input("Enter image name : ")
    ext = int(input("Choose format option 1 or 2 ( 1 -> .jpg , 2 -> .png) : "))
    if ext == 1:
        img.save("./outputs/"+name+".jpg")
        print("Image saved at ./outputs/"+name+".jpg")
    elif ext == 2:
        img.save("./outputs/"+name+".png")
        print("Image saved at ./outputs/"+name+".png")
    else:
        print("Invalid option. Choosing .png as default...")
        img.save("./outputs/"+name+".png")
        print("Image saved at ./outputs/"+name+".png")

# scale image to equal dimensions
def imageScale(lr_image_path):
    path = os.path.splitext(lr_image_path)
    if path[1] != "png":
        foo = Image.open(lr_image_path)
        foo.save(path[0]+".png")
    
    foo = Image.open(path[0]+".png")
    if foo.size[0] > 96*3:
         foo = foo.resize((foo.size[0]//2, foo.size[1]//2), Image.ANTIALIAS)
    foo.save("./rescale/temp.png", optimized=True, quality=95)
    return "./rescale/temp.png"

print("\n**************************************************")
print("\t\tWelcome to Lucid")
image_path = Path(PureWindowsPath(input("Enter Image path : ")))

weights_dir = f'weights/edsr-16-x4'
weights_file = os.path.join(weights_dir, 'weights.h5')

os.makedirs(weights_dir, exist_ok=True)

model = edsr(scale=4, num_res_blocks=16)
model.load_weights(weights_file)

new_path = imageScale(image_path)
generateSR(new_path)

print("\n**************************************************")
