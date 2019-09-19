import os
from model.edsr import edsr
import matplotlib.pyplot as plt
from model import resolve_single
from PIL import Image
import numpy as np

# generate super resolution image
def generateSR(lr_image_path, aspect=1.0):
    lr = np.array(Image.open(lr_image_path))
    sr = resolve_single(model, lr)
    # plt.imshow(sr)
    # plt.savefig(input("Enter image name:")+".png")
    img = Image.fromarray(sr.numpy(), 'RGB')
    n = img.size
    img.thumbnail((round(n[0] * aspect), n[1]), Image.ANTIALIAS)
    img.resize((round(n[0] * aspect), n[1]), Image.ANTIALIAS)
    img.save("./outputs/"+input("Enter image name")+".jpg")
    img.show()

# scale image to equal dimensions
def imageScale(lr_image_path):
    path = os.path.splitext(lr_image_path)
    if path[1] is not ".png":
        foo = Image.open(lr_image_path)
        foo.save(path[0]+".png")
    
    foo = Image.open(path[0]+".png")
    aspect = float(foo.size[0]) / foo.size[1]
    # if foo.size[0] > 96*3:
    #     foo = foo.resize((96*3, 96*3), Image.ANTIALIAS)
    foo.save("./rescale/"+path[0]+".png", optimized=True, quality=95)
    return "./rescale/"+path[0]+".png", aspect

print("\n**************************************************")
print("\t\tWelcome to Lucid")
image_path = input("Enter Image path in unicode format : ")

# Number of residual blocks
depth = 16

# Super-resolution factor
scale = 4

# Downgrade operator
downgrade = 'bicubic'

weights_dir = f'weights/edsr-{depth}-x{scale}'
weights_file = os.path.join(weights_dir, 'weights.h5')

os.makedirs(weights_dir, exist_ok=True)

model = edsr(scale=scale, num_res_blocks=depth)
model.load_weights(weights_file)

new_path, aspect = imageScale(image_path)
generateSR(new_path, aspect)

print("\n**************************************************")
