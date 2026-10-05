import os

from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


OUTPUT_DIR = 'plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def quantifier(img, n_levels):
    step = 256 / n_levels
    q = np.round(img / step) * step
    return np.clip(q, 0, 255).astype(np.uint8)


# Load image
image = Image.open('bmw.jpg').convert('RGB')
I = np.array(image)

# Part 1
print('Shape :', I.shape)
print('Dtype :', I.dtype)
H, W, C = I.shape
print('Hauteur H =', H)
print('Largeur W =', W)
print('Nombre de canaux =', C)

# Part 2 - Quantization
G = np.array(image.convert('L'))
G16 = quantifier(G, 16)
G4 = quantifier(G, 4)

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
for a, im, t in zip(ax, [G, G16, G4], ['256 niveaux', '16 niveaux', '4 niveaux']):
    a.imshow(im, cmap='gray', vmin=0, vmax=255)
    a.set_title(t)
    a.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'quantization.png'), dpi=150)
plt.close()
print('Saved:', os.path.join(OUTPUT_DIR, 'quantization.png'))

# Part 3 - Brightness and histograms
G_claire = np.clip(G.astype(np.int16) + 40, 0, 255).astype(np.uint8)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes[0, 0].imshow(G, cmap='gray', vmin=0, vmax=255)
axes[0, 1].imshow(G_claire, cmap='gray', vmin=0, vmax=255)
axes[1, 0].hist(G.ravel(), bins=256, range=(0, 256))
axes[1, 1].hist(G_claire.ravel(), bins=256, range=(0, 256))

for ax in axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'brightness.png'), dpi=150)
plt.close()
print('Saved:', os.path.join(OUTPUT_DIR, 'brightness.png'))

# Part 5 - Spatial organization
# B: two blocks
# D: checkerboard

y, x = np.indices((8, 8))
B = np.zeros((8, 8), dtype=np.uint8)
B[:, 4:] = 255
D = (((x + y) % 2) * 255).astype(np.uint8)

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes[0, 0].imshow(B, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
axes[0, 1].imshow(D, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
axes[1, 0].hist(B.ravel(), bins=256, range=(0, 256))
axes[1, 1].hist(D.ravel(), bins=256, range=(0, 256))

for ax in axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'spatial.png'), dpi=150)
plt.close()
print('Saved:', os.path.join(OUTPUT_DIR, 'spatial.png'))

print('Image mean:', G.mean())
print('Brighter image mean:', G_claire.mean())
print('B mean:', B.mean(), 'D mean:', D.mean())
