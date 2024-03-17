# logica per convertire immagine in svg
import sys
from PIL import Image, ImageFilter, ImageOps
from potrace import Bitmap, POTRACE_TURNPOLICY_MINORITY  # `potracer` library
import torch.nn as nn
import os
import torch
import matplotlib.pyplot as plt
from torchvision import utils as vutils
import numpy as np
from scipy.ndimage import median_filter


fileName = "outputGlyph"
BATCH_SIZE = 1
LATENT_SIZE = 256


def file_to_svg(filename: str):
    try:
        image = Image.open(filename)
    except IOError:
        print("Image (%s) could not be loaded." % filename)
        return
    bm = Bitmap(image, blacklevel=0.5)
    # bm.invert()
    plist = bm.trace(
        turdsize=2,
        turnpolicy=POTRACE_TURNPOLICY_MINORITY,
        alphamax=1,
        opticurve=False,
        opttolerance=0.2,
    )
    with open(f"{filename}.svg", "w") as fp:
        fp.write(
            f'''<svg version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{image.width}" height="{image.height}" viewBox="0 0 {image.width} {image.height}">''')
        parts = []
        for curve in plist:
            fs = curve.start_point
            parts.append(f"M{fs.x},{fs.y}")
            for segment in curve.segments:
                if segment.is_corner:
                    a = segment.c
                    b = segment.end_point
                    parts.append(f"L{a.x},{a.y}L{b.x},{b.y}")
                else:
                    a = segment.c1
                    b = segment.c2
                    c = segment.end_point
                    parts.append(f"C{a.x},{a.y} {b.x},{b.y} {c.x},{c.y}")
            parts.append("z")
        fp.write(f'<path stroke="none" fill="black" fill-rule="evenodd" d="{"".join(parts)}"/>')
        fp.write("</svg>")




def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

def load_or_create_model(model, model_path, device):
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded from {model_path}")
    else:
        model = to_device(model, device)
        print("New model created")
    return model



#GENERATORE
generator = nn.Sequential(
    # in: latent_size x 1 x 1
    nn.ConvTranspose2d(LATENT_SIZE, 512, kernel_size=4, stride=1, padding=0, bias=False),
    nn.BatchNorm2d(512),
    nn.ReLU(True),
    # out: 512 x 4 x 4

    nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(256),
    nn.ReLU(True),
    # out: 256 x 8 x 8

    nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(True),
    # out: 128 x 16 x 16

    nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(True),
    # out: 64 x 32 x 32

    nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=False),
    nn.Tanh()
    # out: 1 x 64 x 64
)


generator_path = "generator_google.pth"
device = get_default_device()
generator = load_or_create_model(generator, generator_path, device)


# --------------GENERAZIONE IMMAGINE RASTER e SVG ----------------------
# Assicurati che il generatore e i dati siano sulla GPU
# generator = generator.to('cuda')

xb = torch.randn(BATCH_SIZE, LATENT_SIZE, 1, 1)
fake_images = generator(xb)

# Seleziona la prima immagine dal batch
single_fake_image = fake_images[0]

print(type(single_fake_image))
print(single_fake_image.shape)

# Convertila in PNG
# Trasferisci il tensore sulla CPU
tensor_image_cpu = single_fake_image.cpu()
# Converte il tensore in un array NumPy
numpy_image = tensor_image_cpu.mul(255).byte().numpy().squeeze()
original_image = Image.fromarray(numpy_image, mode='L')  # 'L' sta per scala di grigi

original_image = ImageOps.invert(original_image) # Inverti colori
original_image.save(fileName+".png")

# convertila in svg
file_to_svg(fileName+".png")


# --------------APPLICAZIONE FILTRI ----------------------
# Applica un filtro gaussiano
gaussian_filtered_image = original_image.filter(ImageFilter.GaussianBlur(radius=2))

# Applica un filtro mediano
median_filtered_image = original_image.filter(ImageFilter.MedianFilter(size=3))

# Salva le immagini filtrate in formato PNG
gaussian_filtered_image.save(fileName+"_gaussian.png")
median_filtered_image.save(fileName+"_median.png")

# Conversione in svg
file_to_svg(fileName+"_gaussian.png")
file_to_svg(fileName+"_median.png")