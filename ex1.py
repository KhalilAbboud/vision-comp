from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

image = Image.open('bmw.jpg').convert('RGB')
I = np.array(image)

plt.imshow(I)
plt.axis("off")
plt.show()

""" question 1 """
print("Shape :", I.shape)
print("Dtype :", I.dtype)

H, W, C = I.shape
print("Hauteur H =", H)
print("Largeur W =", W)
print("Nombre de canaux =", C)

""" question 2 """
nb_pixels = H * W
print("Nombre total de pixels :", nb_pixels)

""" question 3 """
poids_brut = H * W * 3
print("Poids brut :", poids_brut, "octets")

print("I.nbytes :", I.nbytes, "octets")


