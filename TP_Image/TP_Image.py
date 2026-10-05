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

def negatif(im):
    t=np.array(im)
    h, l, r=t.shape
    for i in range(h):
        for j in range(l):
            for k in range(3):
                t[i,j,k] = 255 - t[i,j,k]
    return Image.fromarray(t)

def cadre_noir(im, ep):
    t = np.array(im)
    h, l, r = t.shape
    for i in range(h):
        for j in range(l):
            if i < ep or i >= h - ep or j < ep or j >= l - ep:
                t[i][j] = [0, 0, 0]
    return Image.fromarray(t)

def rajout_cadre(im, ep):
    t = np.array(im)
    h, l, r = t.shape
    new = np.zeros((h + 2 * ep, l + 2 * ep, 3), dtype="uint8")
    for i in range(h):
        for j in range(l):
            new[i + ep][j + ep] = t[i][j]
    return Image.fromarray(new)



"""alls"""

cocci=Image.open("499p.jpg")
cocci_noire=image_noire(cocci)
cocci_noire.save("cocci_noire.jpg")
cocci_noire.show()

cocci=Image.open("499p.jpg")
cocci_red=composante_rouge(cocci)
cocci_red.save("cocci_red.jpg")
cocci_red.show()

cocci=Image.open("499p.jpg")
cocci_neg=negatif(cocci)
cocci_neg.save("cocci_neg.jpg")
cocci_neg.show()

cocci=Image.open("499p.jpg")
cocci_cadre=cadre_noir(cocci, 50)
cocci_cadre.save("cocci_cadre.jpg")
cocci_cadre.show()

cocci=Image.open("499p.jpg")
cocci_rajout=rajout_cadre(cocci, 50)
cocci_rajout.save("cocci_rajout.jpg")
cocci_rajout.show()