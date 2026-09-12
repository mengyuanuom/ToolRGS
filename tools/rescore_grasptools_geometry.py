"""CPU-only, versioned rescore of all GraspTools test prediction caches.

Source archives/checkpoints are never modified. All original-coordinate V3
rows must have size factor/cap 300 and short side 20; reject mismatched caches.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from utils.grasp_raster import GEOMETRY_VERSION, grasp_matches

GRID=[(i,a) for i in (.25,.5,.75) for a in (5.,10.,20.,30.)]


def summarize(cache):
    n=len(cache['segmentation_iou'])
    sums=np.zeros((12,2),dtype=np.int64)
    for i in range(n):
        m=cache['matches'][cache['match_offsets'][i]:cache['match_offsets'][i+1]]
        for j,(it,at) in enumerate(GRID):
            ok=(m[:,1]>it)&(m[:,2]<=at)
            for q,k in enumerate((1,5)):
                sums[j,q]+=np.any(ok&(m[:,0]<k))
    rates=sums/max(n,1)*100
    return dict(n=n,J1=float(rates[3,0]),J5=float(rates[3,1]),
                mSR1=float(rates[:,0].mean()),mSR5=float(rates[:,1].mean()),
                grid=[dict(iou=i,angle=a,J1=float(rates[j,0]),J5=float(rates[j,1])) for j,(i,a) in enumerate(GRID)])


def process(job):
    source,out,shape,revision=job
    source=Path(source);out=Path(out)
    start=time.time()
    with np.load(source,allow_pickle=False) as a:
        cache={k:a[k] for k in a.files}
    meta=json.loads(str(cache.pop('metadata_json')))
    if meta.get('split')!='test':
        return dict(source=str(source),status='excluded_not_test')
    if meta.get('size_coordinate')!='original' or meta.get('size_factor')!=300 or not np.all(cache['target_width_cap']==300) or not np.all(cache['target_height']==20):
        return dict(source=str(source),status='excluded_geometry_contract',metadata=meta)
    if int(meta.get('max_topk',0))<5:
        return dict(source=str(source),status='excluded_topk_less_than_5',metadata=meta)
    for key in ('rectangles','targets','matches'):
        if not np.isfinite(cache[key]).all():
            raise ValueError(f'Non-finite {key}: {source}')
    old=summarize(cache)
    blocks=[];offsets=[0];n=old['n']
    for i in range(n):
        p=cache['rectangles'][cache['rectangle_offsets'][i]:cache['rectangle_offsets'][i+1]]
        t=cache['targets'][cache['target_offsets'][i]:cache['target_offsets'][i+1]]
        m=np.asarray(grasp_matches(p,t,300.,20.,shape),dtype=np.float32).reshape(-1,3)
        blocks.append(m);offsets.append(offsets[-1]+len(m))
    cache['matches']=np.concatenate(blocks) if blocks else np.empty((0,3),np.float32)
    cache['match_offsets']=np.asarray(offsets,dtype=np.int64)
    cache['image_hw']=np.tile(np.asarray(shape),(n,1))
    new=summarize(cache)
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    target_hash=hashlib.sha256(cache['targets'].tobytes()+cache['target_offsets'].tobytes()).hexdigest()
    pred_hash=hashlib.sha256(cache['rectangles'].tobytes()+cache['rectangle_offsets'].tobytes()).hexdigest()
    label=source.parent.name if source.parent.name!='evaluation_cache' else source.parent.parent.name
    dest=out/(label+'_'+digest[:10]);dest.mkdir(parents=True,exist_ok=True)
    cache['metadata_json']=np.asarray(json.dumps(dict(meta,geometry_version=GEOMETRY_VERSION,
        source_cache_sha256=digest,evaluator_commit=revision),sort_keys=True))
    np.savez_compressed(dest/'predictions_corrected.npz',**cache)
    result=dict(status='complete',source=str(source),metadata=meta,source_sha256=digest,
                prediction_sha256=pred_hash,target_sha256=target_hash,geometry_version=GEOMETRY_VERSION,
                image_hw=shape,evaluator_commit=revision,old=old,corrected=new,
                missing_predictions=int(np.sum(np.diff(cache['rectangle_offsets'])==0)),
                seconds=time.time()-start,output=str(dest))
    (dest/'scores.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--image-shape',type=int,nargs=2,required=True)
    args=parser.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[1],text=True).strip()
    paths=sorted(Path(args.root).rglob('*.npz'))
    jobs=[]
    for p in paths:
        if out.resolve() in p.resolve().parents:
            continue
        with np.load(p,allow_pickle=False) as a:
            if 'metadata_json' not in a.files:
                continue
            m=json.loads(str(a['metadata_json']))
        if m.get('split')=='test':jobs.append((str(p),str(out),tuple(args.image_shape),revision))
    print(f'Found {len(jobs)} test caches',flush=True)
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending={pool.submit(process,j):j[0] for j in jobs}
        for task in as_completed(pending):
            try:r=task.result()
            except Exception as exc:r=dict(source=pending[task],status='error',error=repr(exc))
            results.append(r)
            (out/'summary.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
            print(json.dumps({k:v for k,v in r.items() if k in ('source','status','corrected','error')}),flush=True)
    print('DONE',flush=True)
    if any(r['status']=='error' for r in results):raise SystemExit(1)


if __name__=='__main__':main()
