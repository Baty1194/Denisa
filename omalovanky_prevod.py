#!/usr/bin/env python3
"""
Převod libovolného obrázku na omalovánky podle čísel (A4 PDF).

Použití:
    pip install numpy scipy opencv-python pillow reportlab
    python omalovanky_prevod.py vstup.jpg vystup.pdf
    python omalovanky_prevod.py vstup.jpg vystup.pdf --barev 40 --min-plocha 100 --sum 40

Parametry:
    --barev        počet barev (max 48, výchozí 48)
    --min-plocha   nejmenší plocha v pixelech; menší = hustší mozaika (výchozí 130)
    --sum          síla textury; vyšší = více drobných ploch (výchozí 40, 0 = bez textury)
    --odsum        odšumění vstupu před převodem; pro zrnité fotky zkus 12 (výchozí 0 = vypnuto)
    --font         cesta k TTF fontu s řeckými písmeny (výchozí DejaVuSans)

Použij jen podklady, ke kterým máš práva (vlastní, licencované, volná licence).
Výstup: 1. strana = omalovánka + pás s barvami, 2. strana = hotový obrázek a tabulka barev.
"""
import argparse, os, sys
import numpy as np, cv2
from scipy import ndimage
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

SYMS = list('1234567890') + list('abcdefhkmnpqrtuwxyz') + list('ΔΣΦΨΩΛΠΓΘΞΥΧ') + list('βγδζηθλ')

ap = argparse.ArgumentParser()
ap.add_argument('vstup')
ap.add_argument('vystup')
ap.add_argument('--barev', type=int, default=48)
ap.add_argument('--min-plocha', type=int, default=130)
ap.add_argument('--sum', type=float, default=40)
ap.add_argument('--font', default=None)
ap.add_argument('--odsum', type=float, default=0)
a = ap.parse_args()
K = min(a.barev, len(SYMS))

# --- font ---
cands = [a.font] if a.font else [
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    '/Library/Fonts/DejaVuSans.ttf',
    'C:/Windows/Fonts/DejaVuSans.ttf',
    'DejaVuSans.ttf']
fp = next((p for p in cands if p and os.path.exists(p)), None)
if not fp:
    sys.exit('Nenašel jsem DejaVuSans.ttf, zadej cestu parametrem --font')
bp = fp.replace('DejaVuSans.ttf', 'DejaVuSans-Bold.ttf')
pdfmetrics.registerFont(TTFont('DV', fp))
pdfmetrics.registerFont(TTFont('DVB', bp if os.path.exists(bp) else fp))

# --- načtení a zmenšení na A4 při 300 dpi ---
img = cv2.imread(a.vstup)
if img is None:
    sys.exit(f'Nepodařilo se načíst obrázek: {a.vstup}')
if a.odsum > 0:
    img = cv2.fastNlMeansDenoisingColored(img, None, a.odsum, a.odsum, 7, 21)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h0, w0 = img.shape[:2]
s = min(2300 / w0, 3150 / h0)
img = cv2.resize(img, (int(w0 * s), int(h0 * s)), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
H, W = img.shape[:2]

rg = np.random.default_rng(5)


def nz(scale, octv):
    acc = np.zeros((H, W), np.float32)
    amp = tot = 0.0
    amp = 1.0
    for o in range(octv):
        sc = scale / (2 ** o)
        r = rg.random((max(2, int(H / sc)), max(2, int(W / sc)))).astype(np.float32)
        acc += cv2.resize(r, (W, H), interpolation=cv2.INTER_CUBIC) * amp
        tot += amp
        amp *= 0.5
    return acc / tot - 0.5


if a.sum > 0:
    noisy = img.astype(np.float32) + (nz(26, 2) * a.sum)[..., None] + \
        np.stack([nz(40, 2) * (a.sum / 4) for _ in range(3)], -1)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
else:
    noisy = img
sm = cv2.bilateralFilter(noisy, 5, 18, 4)
lab = cv2.cvtColor(sm, cv2.COLOR_RGB2LAB).astype(np.float32)

# --- kvantizace barev ---
samp = lab[::7, ::7].reshape(-1, 3)
crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
_, _, cent = cv2.kmeans(samp, K, None, crit, 2, cv2.KMEANS_PP_CENTERS)
flat = lab.reshape(-1, 3)
L = np.empty(flat.shape[0], np.int32)
for i in range(0, flat.shape[0], 600000):
    ch = flat[i:i + 600000]
    L[i:i + 600000] = ((ch[:, None, :] - cent[None, :, :]) ** 2).sum(2).argmin(1)
L = L.reshape(H, W)


def comps(L):
    cc = np.zeros(L.shape, np.int32)
    off = 0
    for k in range(K):
        lb, n = ndimage.label(L == k)
        m = lb > 0
        cc[m] = lb[m] + off
        off += n
    return cc, off


for _ in range(4):
    cc, n = comps(L)
    small = np.bincount(cc.ravel(), minlength=n + 1)[cc] < a.min_plocha
    if not small.any() or small.all():
        break
    idx = ndimage.distance_transform_edt(small, return_distances=False, return_indices=True)
    L = np.where(small, L[idx[0], idx[1]], L)
cc, n = comps(L)
print('ploch:', n)

# --- paleta ---
cnt = np.bincount(L.ravel(), minlength=K).astype(np.float64)
pal = np.zeros((K, 3), np.float32)
for c_ in range(3):
    pal[:, c_] = np.bincount(L.ravel(), weights=sm[..., c_].ravel().astype(np.float64), minlength=K) / np.maximum(cnt, 1)
used = [k for k in range(K) if cnt[k] > 0]
lum = {k: 0.299 * pal[k, 0] + 0.587 * pal[k, 1] + 0.114 * pal[k, 2] for k in used}
order = sorted(used, key=lambda k: lum[k])
sym_of = {k: SYMS[i] for i, k in enumerate(order)}

# --- obrysy ---
b = np.zeros((H, W), bool)
b[:, :-1] |= cc[:, :-1] != cc[:, 1:]
b[:-1, :] |= cc[:-1, :] != cc[1:, :]
b = cv2.dilate(b.astype(np.uint8), np.ones((2, 2), np.uint8)) > 0
art = np.full((H, W), 255, np.uint8)
art[b] = 105
art[:3, :] = art[-3:, :] = art[:, :3] = art[:, -3:] = 60
Image.fromarray(art).save('_art_tmp.png', optimize=True)

# --- popisky ---
labels = []
for i, sl in enumerate(ndimage.find_objects(cc), start=1):
    if sl is None:
        continue
    comp = cc[sl] == i
    area = int(comp.sum())
    edt = ndimage.distance_transform_edt(np.pad(comp, 1))[1:-1, 1:-1]
    placed = 0
    while placed < 6:
        mx = edt.max()
        if mx < (5.5 if placed == 0 else 38.0):
            break
        iy, ix = np.unravel_index(np.argmax(edt), edt.shape)
        ry, rx = sl[0].start + iy, sl[1].start + ix
        labels.append((rx, ry, sym_of[int(L[ry, rx])], float(mx)))
        placed += 1
        if area < 26000:
            break
        yy, xx = np.ogrid[:edt.shape[0], :edt.shape[1]]
        edt[(yy - iy) ** 2 + (xx - ix) ** 2 < 150 ** 2] = 0

# --- PDF ---
PW, PH = A4
kp = 72.0 / 300.0
AWp, AHp = W * kp, H * kp
ox, oy = (PW - AWp) / 2, 62
c = canvas.Canvas(a.vystup, pagesize=A4)
c.drawImage(ImageReader(Image.open('_art_tmp.png')), ox, oy, AWp, AHp)
c.setFillColorRGB(0.18, 0.18, 0.18)
for x, y, sy, r in labels:
    fs = max(3.0, min(8.0, r / 300 * 25.4 * 5.2))
    c.setFont('DV', fs)
    c.drawCentredString(ox + x * kp, oy + (H - y) * kp - fs * 0.35, sy)
sw = AWp / len(order)
for i, k in enumerate(order):
    x0 = ox + i * sw
    c.setFillColorRGB(*(pal[k] / 255.0))
    c.setStrokeColorRGB(0.2, 0.2, 0.2)
    c.setLineWidth(0.4)
    c.rect(x0 + 0.8, 28, sw - 1.6, 26, fill=1, stroke=1)
    c.setFillColorRGB(*((1, 1, 1) if lum[k] < 120 else (0, 0, 0)))
    c.setFont('DVB', 9)
    c.drawCentredString(x0 + sw / 2, 37, sym_of[k])
c.showPage()

ref = Image.fromarray(pal[L].astype(np.uint8)).resize((W // 2, H // 2), Image.NEAREST)
ref.save('_ref_tmp.jpg', quality=88)
c.setFont('DVB', 16)
c.setFillColorRGB(0.1, 0.1, 0.1)
c.drawCentredString(PW / 2, 810, 'Hotový obrázek a paleta barev')
rows = (len(order) + 3) // 4
rh = min(360 * H / W, 800 - 10 - 40 - 18 - rows * 17 - 20)  # aby se tabulka vešla na stránku
rw = rh * W / H
c.drawImage(ImageReader('_ref_tmp.jpg'), (PW - rw) / 2, 800 - rh - 10, rw, rh)
ty = 800 - rh - 40
c.setFont('DVB', 9)
c.drawString(40, ty, 'Paleta (symbol, RGB, HEX):')
for i, k in enumerate(order):
    col, row = i % 4, i // 4
    x, y = 40 + col * 128, ty - 18 - row * 17
    c.setFillColorRGB(*(pal[k] / 255.0))
    c.setStrokeColorRGB(0.2, 0.2, 0.2)
    c.rect(x, y - 3, 14, 12, fill=1, stroke=1)
    c.setFillColorRGB(0.1, 0.1, 0.1)
    c.setFont('DVB', 9)
    c.drawString(x + 19, y, sym_of[k])
    r_, g_, b_ = [int(round(v)) for v in pal[k]]
    c.setFont('DV', 6.5)
    c.drawString(x + 32, y, f'{r_},{g_},{b_}  #{r_:02X}{g_:02X}{b_:02X}')
c.showPage()
c.save()
os.remove('_art_tmp.png')
os.remove('_ref_tmp.jpg')
print('Hotovo:', a.vystup)
