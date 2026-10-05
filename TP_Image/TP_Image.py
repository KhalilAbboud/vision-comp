from PIL import Image
import numpy as np

def image_noire(im):
    t=np.array(im)
    h, l, r=t.shape
    for i in range(h):
        for j in range(l):
            for k in range(3):
                t[i,j,k]=0
    return Image.fromarray(t)

def composante_rouge(im):
    t=np.array(im)
    h, l, r=t.shape
    for i in range(h):
        for j in range(l):
            t[i,j,1]=0
            t[i,j,2]=0
    return Image.fromarray(t)


"""alls"""

cocci=Image.open("499p.jpg")
cocci_noire=image_noire(cocci)
cocci_noire.save("cocci_noire.jpg")
cocci_noire.show()

cocci=Image.open("499p.jpg")
cocci_red=composante_rouge(cocci)
cocci_red.save("cocci_red.jpg")
cocci_red.show()