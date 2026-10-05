from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def quantifier(img, n_levels):
    step = 256 / n_levels
    q = np.round(img / step) * step
    return np.clip(q, 0, 255).astype(np.uint8)


image = Image.open('bmw.jpg').convert('RGB')
I = np.array(image)

plt.imshow(I)
plt.axis("off")
plt.show()

""" Part 1 """
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


"""
Une image RGB a la forme (hauteur, largeur, 3) car chaque pixel contient trois canaux : rouge, vert et bleu. Elle n’a pas trois fois plus de pixels qu’une image en niveaux de gris de même taille, mais elle utilise trois fois plus de données par pixel. 
Enfin, le fichier JPEG peut être beaucoup plus petit que le poids brut car le format JPEG utilise une compression pour réduire la taille du fichier.
"""


"""Partie2"""
"""A-Resolution spatial"""
H, W = I.shape[:2]
largeur_cm = W / 300 * 2.54
hauteur_cm = H / 300 * 2.54
print(largeur_cm, hauteur_cm)

"""B-Quantization"""
G = np.array(image.convert("L"))
G16 = quantifier(G, 16)
G4  = quantifier(G, 4)

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
for a, im, t in zip(ax, [G, G16, G4], ["256 niveaux", "16 niveaux", "4 niveaux"]):
    a.imshow(im, cmap="gray", vmin=0, vmax=255); a.set_title(t); a.axis("off")
plt.show()

print(G.nbytes, G16.nbytes, G4.nbytes)

"""
response:
A. Résolution spatiale. Quand la résolution passe de 300 à 150 ppp avec le même nombre de pixels, la taille d'impression double dans chaque dimension (et est multipliée par 4 en surface). Le nombre de pixels de l'image ne change pas, c'est une propriété du fichier. La densité d'impression (pixels par pouce) est une propriété de la sortie : elle diminue, donc l'impression est plus grande mais moins fine.

B. Résolution en luminance.

2-Nombre de bits par pixel = log₂(niveaux) : 256 niveaux → 8 bits, 16 niveaux → 4 bits, 4 niveaux → 2 bits.

3-Les dégradés doux (ciel, ombres) et les textures fines deviennent moins visibles.
On observe des paliers (faux contours) et des zones aplaties, de type « posterisation », surtout à 4 niveaux.

4-Les trois tableaux ont le même nbytes (H × W octets).
Ils restent de type uint8, donc un octet par pixel quel que soit le nombre de valeurs distinctes.
Réduire le nombre de niveaux ne réduit la mémoire que si l'on empaquette les bits (par exemple 4 pixels par octet à 2 bits) ou si l'on compresse.
"""



"""Partie3"""
"""Histogrammes et luminosité"""
# 1. Gray histogram
plt.hist(G.ravel(), bins=256, range=(0, 256))
plt.xlabel("Intensité"); plt.ylabel("Nombre de pixels"); plt.show()

# 2. RGB histograms
for c, name, col in zip(range(3), ["Rouge", "Vert", "Bleu"], ["r", "g", "b"]):
    plt.hist(I[:, :, c].ravel(), bins=256, range=(0, 256), color=col, alpha=0.6, label=name)
plt.legend(); plt.show()

# 3. Mean luminosity
L = G.sum() / (G.shape[0] * G.shape[1])   # == G.mean()

# 4-5. Brighter image
G_claire = np.clip(G.astype(np.int16) + 40, 0, 255).astype(np.uint8)
fig, ax = plt.subplots(2, 2, figsize=(12, 8))
ax[0,0].imshow(G, cmap="gray", vmin=0, vmax=255)
ax[0,1].imshow(G_claire, cmap="gray", vmin=0, vmax=255)
ax[1,0].hist(G.ravel(), bins=256, range=(0,256))
ax[1,1].hist(G_claire.ravel(), bins=256, range=(0,256))
plt.show()
print(G.mean(), G_claire.mean())

"""Reponse:"""
"""
a. Une concentration près de 0 indique des zones sombres (sous-exposition, ombres écrasées). Près de 255, elle indique des zones très claires ou saturées (surexposition, perte de détails).
b. En uint8, le calcul se fait modulo 256 : 250 + 40 = 34, donc des pixels clairs deviendraient sombres. Avec int16, la somme est représentable. np.clip la ramène ensuite à 255 avant la reconversion en uint8.
c. Non. Tout pixel avec G > 215 est écrêté à 255 et gagne donc moins de 40. La moyenne n'augmente de 40 exactement que si aucun pixel ne dépasse 215. Sinon, l'augmentation est strictement inférieure à 40.
d. Non. L'histogramme compte les intensités mais ignore les positions des pixels. Il indique combien de pixels sont sombres, pas où ils se trouvent.
e. Pas nécessairement. Ajouter 40 décale l'histogramme vers la droite sans modifier son étalement, donc le contraste reste le même.
L'écrêtage le réduit même dans les zones claires : des pixels distincts (par exemple 220 et 250) deviennent tous égaux à 255. 
"""


"""Partie4"""
"""Cooccurrence"""
"""
First row pairs: (0,0), (0,1), (1,1). Each row has 3 pairs, so 4 × 3 = 12 pairs total.
Manual C, with all 12 pairs counted:
     j=0 j=1 j=2 j=3
i=0 [ 1   2   0   0 ]
i=1 [ 0   2   1   0 ]
i=2 [ 0   0   1   2 ]
i=3 [ 0   0   0   3 ]
"""

A = np.array([
    [0, 0, 1, 1],
    [0, 1, 1, 2],
    [2, 2, 3, 3],
    [3, 3, 3, 3],
], dtype=np.uint8)

"""3"""
C = np.zeros((4, 4), dtype=int)
for y in range(A.shape[0]):
    for x in range(A.shape[1] - 1):
        i = A[y, x]
        j = A[y, x + 1]
        C[i, j] += 1          
P = C / C.sum()
print(C, P.sum())

"""4-5"""
"""
P = C / 12, and the sum of P is 1.
Energy: E = (1+4+4+1+1+4+9)/144 = 24/144 = 1/6 ≈ 0.167.
Contrast: only off-diagonal terms count. (0,1)×2, (1,2)×1, (2,3)×2 each have (i−j)² = 1, so K = 5/12 ≈ 0.417.
"""
i_idx, j_idx = np.indices(P.shape)
E = (P**2).sum()
K = ((i_idx - j_idx)**2 * P).sum()

"""6-7"""
"""
Valeurs à reporter :

Couples de la première ligne : (0,0), (0,1), (1,1). Chaque ligne a 3 couples, donc 4 × 3 = 12 couples au total.
P = C / 12, donc ΣP = 1.
E = 24/144 = 1/6 ≈ 0,167 et K = 5/12 ≈ 0,417.
Image uniforme (tous les pixels à 2) : C[2,2] = 12, donc P[2,2] = 1, E = 1 et K = 0.

7-Les éléments diagonaux C[i,i] comptent les couples de voisins ayant le même niveau de gris, c'est-à-dire les zones homogènes.
Pour l'image uniforme, tous les couples vérifient i = j, donc (i − j)² = 0 et le contraste est nul.
Une énergie égale à 1 signifie que toute la probabilité est concentrée dans une seule case de la matrice. Il n'y a qu'un seul type de transition : l'image est parfaitement uniforme et régulière, sans variation de texture.
"""


"""Partie5"""
"""Histogramme et organisation spatiale"""
y, x = np.indices((8, 8))
B = np.zeros((8, 8), dtype=np.uint8); B[:, 4:] = 255
D = (((x + y) % 2) * 255).astype(np.uint8)

fig, ax = plt.subplots(2, 2, figsize=(10, 8))
ax[0,0].imshow(B, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
ax[0,1].imshow(D, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
ax[1,0].hist(B.ravel(), bins=256, range=(0,256))
ax[1,1].hist(D.ravel(), bins=256, range=(0,256))
plt.show()
print(B.mean(), D.mean())   # 127.5 and 127.5


"""Respones"""
"""
Textures et transitions : B est composée de deux blocs homogènes (noir à gauche, blanc à droite) avec une seule frontière verticale. D est un damier fin où tous les voisins immédiats diffèrent.
Pourquoi les histogrammes sont identiques : B et D contiennent chacune 32 pixels à 0 et 32 pixels à 255, donc la même luminosité moyenne (127,5). L'histogramme ne tient pas compte de la position des pixels.
Information manquante : l'organisation spatiale, c'est-à-dire la disposition des valeurs les unes par rapport aux autres.
Changements brusques les plus fréquents : dans D. Les 112 paires de voisins horizontaux et verticaux (56 + 56) passent de 0 à 255 ou inversement. Dans B, il n'y en a que 8, le long de la frontière centrale.
Lien avec les contours : un contour est une zone de changement brusque d'intensité. B présente un seul contour net et localisé. D en présente partout, ce qui correspond à une texture de haute fréquence. L'histogramme seul ne permet de distinguer ni l'un ni l'autre.
"""
