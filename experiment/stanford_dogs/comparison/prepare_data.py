"""Download official archives, audit splits/images, and validate the actual CPU loader."""
import collections,hashlib,json,os
from pathlib import Path
import sys,tarfile,time,urllib.request
ROOT=Path(__file__).resolve().parent
PLAN=json.loads((ROOT/'plan.json').read_text())
DATA=ROOT/'data/StanfordDogs'
assert not (ROOT/'data-validation.json').exists()
DATA.mkdir(parents=True,exist_ok=False)
archives={}
EXPECTED={'images':'a443fc2f9a851aa1a7b8e4fa4af94b8459fce2c9f5a2f4dd4669cf9673a4ec58','annotation':'662ed36f707be6b195c2fda392d8e153e83af28cedc8e2134a649f198def30fd','lists':'34b47cacd9a98b5d150e084f24d29391c084c55272295ec65c85651bc35f4d6c'}
for name in ['images','annotation','lists']:
    url='http://vision.stanford.edu/aditya86/ImageNetDogs/'+name+'.tar'
    target=DATA/(name+'.tar')
    h=hashlib.sha256(); count=0; last=time.monotonic()
    with urllib.request.urlopen(url,timeout=60) as response,target.open('xb') as out:
        expected=int(response.headers['Content-Length'])
        while True:
            block=response.read(1024*1024)
            if not block:break
            out.write(block);h.update(block);count+=len(block)
            if time.monotonic()-last>15:
                print(json.dumps({'event':'download','archive':name,'bytes':count,'total':expected}),flush=True);last=time.monotonic()
    assert count==expected
    assert h.hexdigest()==EXPECTED[name], "Official TFDS checksum mismatch"
    archives[name]={'url':url,'bytes':count,'sha256':h.hexdigest()}
    print(json.dumps({'event':'downloaded','archive':name,**archives[name]}),flush=True)
    with tarfile.open(target) as archive:
        members=archive.getmembers()
        for member in members:
            destination=(DATA/member.name).resolve()
            assert destination.is_relative_to(DATA.resolve()) and not member.issym() and not member.islnk()
        archive.extractall(DATA,members=members)
(ROOT/'download-manifest.json').write_text(json.dumps(archives,indent=2))
import numpy as np
from scipy.io import loadmat
from PIL import Image
splits={};class_folders={};counts={}
for split,expected in [('train',12000),('test',8580)]:
    m=loadmat(DATA/(split+'_list.mat'))
    files=[str(v[0][0]) for v in m['annotation_list']]
    labels=m['labels'].reshape(-1).astype(int)-1
    assert len(files)==len(set(files))==len(labels)==expected
    assert set(labels)==set(range(120))
    counter=collections.Counter(int(x) for x in labels)
    if split=='train':assert set(counter.values())=={100}
    for i,(name,label) in enumerate(zip(files,labels)):
        folder=name.split('/')[0]
        if int(label) in class_folders:assert class_folders[int(label)]==folder
        class_folders[int(label)]=folder
        path=DATA/'Images'/(name+'.jpg')
        assert path.is_file() and (DATA/'Annotation'/name).is_file()
        with Image.open(path) as image:image.verify()
        if i%4000==0:print(json.dumps({'event':'image_audit','split':split,'checked':i}),flush=True)
    splits[split]=set(files);counts[split]=dict(counter)
assert not splits['train']&splits['test']
assert len(set(class_folders.values()))==120
import torch
from unittest.mock import patch
sys.path.insert(0,str(ROOT/'src'))
from data.dataloaders import get_dataset
assert not torch.cuda.is_available()
original=torch.tensor
def cpu_tensor(*args,**kwargs):
    if kwargs.get('device')=='cuda':kwargs['device']='cpu'
    return original(*args,**kwargs)
with patch('torch.tensor',side_effect=cpu_tensor):
    train,test=get_dataset('stanforddogs',256,224,'random',str(ROOT/'data'))
assert len(train)==12000 and len(test)==8580 and train.num_classes==test.num_classes==120
for dataset,shape in [(train,(3,256,256)),(test,(3,224,224))]:
    image,label=dataset[0]
    assert image.shape==shape and image.dtype==torch.float32
    assert 0<=image.min()<=image.max()<=1 and 0<=label<120
    assert torch.isfinite(dataset.normalize(image[None])).all()
from config import DistillCfg,EvalCfg
for mode in ['pixel','generator_pixel']:
    cfg=DistillCfg(explicit_bool=True).parse_args(['--dataset=stanforddogs','--model=dinov2_vitb','--distill_mode='+mode,'--distill_opt=adam','--lr=0.002','--iterations=5000','--ipc=1','--num_workers=2','--augs_per_batch=1'])
    assert type(cfg.lr) is float and cfg.lr==.002
    for key in ['iterations','ipc','num_workers','augs_per_batch']:assert type(getattr(cfg,key)) is int
result=dict(status='passed',classes=120,train_images=12000,test_images=8580,total_images=20580,all_image_headers_verified=True,train_test_paths_disjoint=True,labels='int64 0..119',class_folders=class_folders,class_names=train.class_names,class_counts=counts,actual_cpu_loader=True,bbox_crop=False,archives=archives)
with (ROOT/'data-validation.json').open('x') as out:json.dump(result,out,indent=2)
print(json.dumps({'event':'data_validation','status':'passed','train':len(train),'test':len(test)}),flush=True)
