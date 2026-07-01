"""Turn Dr. K's painted marks (in RIT_ground_truth/to_mark/*.png) into binary ground-truth masks.

Colours (opaque): IRIS = pure RED, SCLERA = pure BLUE, ALMOND = optional pure GREEN outline (else the
almond is iris + sclera). For each MARKED image, writes:
  masks/<id>_iris.png  masks/<id>_sclera.png  masks/<id>_almond.png   (binary 0/255)
  overlays/<id>.png     (raw crop with the three masks tinted, for a quick look)
  polygons.json         (mask outlines as polygons)
Unmarked images (still clean) are skipped. Nothing tuned or committed.

Usage: py -3.14 tools/rit_extract_ground_truth.py [out_dir]
"""
import json, sys
from pathlib import Path
import cv2, numpy as np

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "outputs/nystagmus at rest to left in right vestibular neuritis_tracked")
GT = OUT / "RIT_ground_truth"
TM = GT / "to_mark"; RAW = GT / "raw"
(GT / "masks").mkdir(exist_ok=True); (GT / "overlays").mkdir(exist_ok=True)


def color_masks(img):
    # PURE painted colours only (MS-Paint palette red/blue/green) -- strict, so natural reddish
    # conjunctiva / skin never counts as a mark.
    B, G, R = img[:, :, 0].astype(int), img[:, :, 1].astype(int), img[:, :, 2].astype(int)
    red = ((R > 180) & (G < 70) & (B < 70)).astype(np.uint8) * 255                       # IRIS
    blue = ((B > 170) & (R < 80) & (G < 100)).astype(np.uint8) * 255                     # SCLERA
    green = ((G > 170) & (R < 80) & (B < 100)).astype(np.uint8) * 255                    # ALMOND outline
    return red, blue, green


def fill_outline(green):
    if int((green > 0).sum()) < 30:
        return None
    g = cv2.morphologyEx(green, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    cnts, _ = cv2.findContours(g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    m = np.zeros(green.shape, np.uint8)
    cv2.drawContours(m, [max(cnts, key=cv2.contourArea)], -1, 255, -1)
    return m


def polys(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c.reshape(-1, 2).tolist() for c in cnts if cv2.contourArea(c) > 20]


poly_out = {}
done = 0
for f in sorted(TM.glob("*.png")):
    uid = f.stem
    img = cv2.imread(str(f))
    red, blue, green = color_masks(img)
    if int((red > 0).sum()) < 120 or int((blue > 0).sum()) < 120:
        continue                                             # a real mark has BOTH iris-red AND sclera-blue
    iris = cv2.morphologyEx(red, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    sclera = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    almond = fill_outline(green)
    if almond is None:                                               # derive almond = iris + sclera
        almond = cv2.bitwise_or(iris, sclera)
        almond = cv2.morphologyEx(almond, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    cv2.imwrite(str(GT / "masks" / f"{uid}_iris.png"), iris)
    cv2.imwrite(str(GT / "masks" / f"{uid}_sclera.png"), sclera)
    cv2.imwrite(str(GT / "masks" / f"{uid}_almond.png"), almond)
    base = cv2.imread(str(RAW / f"{uid}.png")) if (RAW / f"{uid}.png").exists() else img
    ov = base.copy()
    ov[almond > 0] = (0.7 * ov[almond > 0] + np.array([0, 120, 0])).astype(np.uint8)
    ov[sclera > 0] = (0.5 * ov[sclera > 0] + np.array([200, 90, 0])).astype(np.uint8)
    ov[iris > 0] = (0.5 * ov[iris > 0] + np.array([0, 0, 200])).astype(np.uint8)
    cv2.imwrite(str(GT / "overlays" / f"{uid}.png"), ov)
    poly_out[uid] = dict(iris=polys(iris), sclera=polys(sclera), almond=polys(almond))
    done += 1
json.dump(poly_out, open(GT / "polygons.json", "w", encoding="utf-8"), indent=1)
print(f"extracted {done} marked units -> {GT/'masks'} (+ overlays, polygons.json)")
if done == 0:
    print("No marked images found. Paint the masks in", TM, "then re-run (see MARKING_INSTRUCTIONS.md).")
