#!/usr/bin/env python3
"""Break down recapture sub-signals (lap, teng, resid, hf, face_px) per image."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2, numpy as np
from app.config import Settings, resolve_device
from app.face_detect import FaceDetector
from app.multisignal import _ramp

settings = Settings()
detector = FaceDetector(settings.MODEL_DIR)

def raw_components(face_native_bgr, face_px):
    g = cv2.cvtColor(cv2.resize(face_native_bgr, (224, 224)), cv2.COLOR_BGR2GRAY)
    gf = g.astype(np.float32)
    lap = float(cv2.Laplacian(g, cv2.CV_64F).var())
    gx = cv2.Sobel(gf, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gf, cv2.CV_32F, 0, 1)
    teng = float(np.sqrt(gx*gx+gy*gy).mean())
    resid = float((gf - cv2.GaussianBlur(gf, (3,3), 0)).std())
    spec = np.fft.fftshift(np.abs(np.fft.fft2(gf)))
    yy, xx = np.mgrid[0:224,0:224]
    dist = np.sqrt((yy-112)**2+(xx-112)**2)
    hf = float(spec[dist>60].sum()/(spec.sum()+1e-9))
    s_lap=_ramp(lap,800.,2600.); s_teng=_ramp(teng,25.,68.); s_resid=_ramp(resid,3.,10.); s_hf=_ramp(hf,0.15,0.42)
    detail = 0.45*s_lap+0.20*s_teng+0.20*s_resid+0.15*s_hf
    s_res = _ramp(float(face_px), 20000., 45000.)
    final = min(1.0, 0.80*detail+0.20*s_res)
    return dict(lap=lap,teng=teng,resid=resid,hf=hf,s_lap=s_lap,s_teng=s_teng,s_resid=s_resid,s_hf=s_hf,detail=detail,face_px=face_px,s_res=s_res,final=final)

CAL = Path("/home/mrnurali/E-GAZ/docs/plans/calibration/incident_urgut")
sets = {
    "REAL": sorted((CAL/"original").glob("*.jpg")),
    "SPOOF": sorted((CAL/"urgut").glob("*.jpg")),
}

hdr = f"{'SET':<7}{'file':<32}{'lap':>8}{'teng':>7}{'resid':>7}{'hf':>7}{'s_lap':>7}{'s_teng':>7}{'s_res':>7}{'s_hf':>7}{'fpx':>9}{'s_fpx':>7}{'final':>7}"
print(hdr)
for label, files in sets.items():
    for f in files:
        img = cv2.imread(str(f))
        bbox = detector.detect(img)
        if bbox is None:
            print(f"{label:<7}{f.name:<32} no_face"); continue
        from app.liveness import _crop_face
        crop = _crop_face(img, bbox, None, 224, 224)
        face_px = max(0,bbox[2])*max(0,bbox[3])
        c = raw_components(crop, face_px)
        print(f"{label:<7}{f.name:<32}{c['lap']:>8.1f}{c['teng']:>7.2f}{c['resid']:>7.2f}{c['hf']:>7.3f}{c['s_lap']:>7.2f}{c['s_teng']:>7.2f}{c['s_resid']:>7.2f}{c['s_hf']:>7.2f}{c['face_px']:>9}{c['s_res']:>7.2f}{c['final']:>7.3f}")
