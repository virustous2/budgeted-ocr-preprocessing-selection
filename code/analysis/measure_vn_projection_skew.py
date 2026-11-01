"""Diagnostic projection-profile skew estimates; does not change OCR or features."""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def angle_grid(bound=5.0, step=0.05):
    if bound <= 0 or step <= 0 or not np.isclose(2*bound/step, round(2*bound/step)):
        raise ValueError('Range must be positive and an exact multiple of step')
    return np.linspace(-bound, bound, round(2*bound/step)+1)


def estimate(gray, angles, max_width=1000):
    if gray.ndim != 2 or not gray.size:
        raise ValueError('Expected nonempty grayscale image')
    if gray.shape[1] > max_width:
        gray = cv2.resize(gray, (max_width, round(gray.shape[0]*max_width/gray.shape[1])), interpolation=cv2.INTER_AREA)
    _, ink = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # Keep the whole rotated page: zero padding prevents angle-dependent clipping.
    h,w=ink.shape; side=int(np.ceil(np.hypot(h,w)))+4
    canvas=np.zeros((side,side),np.float32)
    y,x=(side-h)//2,(side-w)//2;canvas[y:y+h,x:x+w]=ink
    scores=[]
    for a in angles:
        m=cv2.getRotationMatrix2D(((side-1)/2,(side-1)/2),float(a),1)
        rotated=cv2.warpAffine(canvas,m,(side,side),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
        profile=rotated.sum(axis=1,dtype=np.float64)
        scores.append(float(np.square(np.diff(profile)).sum()))
    scores=np.asarray(scores)
    # Equal maxima prefer smallest absolute correction, then smaller signed angle.
    best=min(range(len(angles)),key=lambda i:(-scores[i],abs(angles[i]),angles[i]))
    correction=float(angles[best])
    return correction,scores,dict(flat_objective=bool(np.ptp(scores)<1e-12),
        boundary_maximum=best in (0,len(angles)-1),ink_fraction=float(ink.mean()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--limit',type=int,default=300)
    p.add_argument('--max-width',type=int,default=1000)
    p.add_argument('--bound',type=float,default=5)
    p.add_argument('--step',type=float,default=.05)
    args=p.parse_args()
    cv2.setNumThreads(2)
    if not 1<=args.limit<=300 or args.max_width<100:raise ValueError('Invalid limit/width')
    manifest=ROOT/'outputs/publication_v3_casefold_20261010_v2/vietnamese_cohort_manifest.csv'
    with manifest.open(newline='') as f:cohort=list(csv.DictReader(f))
    if len(cohort)!=300 or len({r['image_id'] for r in cohort})!=300:raise ValueError('Cohort mismatch')
    jobs=cohort[:args.limit];angles=angle_grid(args.bound,args.step)
    # Validate all selected raw inputs before creating any output.
    for r in jobs:
        path=ROOT/'datasets/300_images/300_images'/r['image_file']
        if sha(path)!=r['image_sha256']:raise ValueError('Image hash mismatch: '+r['image_id'])
    args.output_dir.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();rows=[]
    with (args.output_dir/'angle_scores.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['image_id','correction_deg','projection_score'])
        for r in jobs:
            path=ROOT/'datasets/300_images/300_images'/r['image_file']
            gray=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
            if gray is None:raise ValueError('Unreadable image: '+str(path))
            a,scores,flags=estimate(gray,angles,args.max_width)
            rows.append(dict(image_id=r['image_id'],image_sha256=r['image_sha256'],
                correction_deg=a,absolute_estimate_deg=abs(a),**flags))
            writer.writerows((r['image_id'],float(a),float(s)) for a,s in zip(angles,scores));f.flush()
            print(f"{len(rows)}/{len(jobs)} {r['image_id']} correction={a:.2f}",flush=True)
    with (args.output_dir/'per_page.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    values=np.array([r['absolute_estimate_deg'] for r in rows])
    summary=dict(scope='raw only; diagnostic, not ground-truth skew or deskew benefit',
        complete_cohort=len(rows)==300,pages=len(rows),median_abs_deg=float(np.median(values)),
        max_abs_deg=float(values.max()),above_1_deg=int((values>1+1e-9).sum()),above_2_deg=int((values>2+1e-9).sum()),
        flags=[r for r in rows if r['flat_objective'] or r['boundary_maximum']],
        seconds=time.monotonic()-start,config=vars(args)|{'output_dir':str(args.output_dir)},
        script_sha256=sha(Path(__file__)),cohort_sha256=sha(manifest),opencv=cv2.__version__,opencv_threads=2,
        files={name:sha(args.output_dir/name) for name in ['per_page.csv','angle_scores.csv']})
    (args.output_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
