"""Independent CPU validation; neither optimizes images nor submits jobs."""
import csv,hashlib,json,math,os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
PLAN=json.loads((ROOT/'experiment/plan.json').read_text())
sys.path.insert(0,str(ROOT/'src'))
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 assert os.environ['CUDA_VISIBLE_DEVICES']==''
 import torch
 from PIL import Image
 from config import DistillCfg
 torch.set_num_threads(2)
 assert not torch.cuda.is_available()
 s=json.loads((ROOT/'status.json').read_text())
 assert s['status']=='success' and s['completed_updates']==5001
 for relative,expected in json.loads((ROOT/'experiment/source-manifest.json').read_text())['files'].items():assert sha(ROOT/relative)==expected,relative
 imageset=torch.load(s['data_path'],map_location='cpu',weights_only=True)
 images,labels=imageset['images'],imageset['labels']
 assert images.shape==(120,3,256,256) and images.dtype==torch.float32
 assert torch.equal(labels,torch.arange(120)) and labels.dtype==torch.int64
 assert torch.isfinite(images).all() and 0<=images.min()<=images.max()<=1
 assert sha(s['data_path'])==s['data_sha256']
 ck=torch.load(ROOT/'checkpoints/step_05000_final.pth',map_location='cpu',weights_only=False)
 assert ck['global_step']==5000
 package=ck['synset']
 assert all(int(v['step'].item())==5001 for v in package['opt_state']['state'].values())
 cfg=DistillCfg(explicit_bool=True).parse_args(PLAN['cli_args'])
 difference=None
 if PLAN['mode']=='generator':
  from synsets.generator import GeneratorDataset
  from run_distill import tensor_hash
  assert s['fixed_noise_unchanged'] and s['generator_parameters_updated'] and s['adam_steps']==5001
  assert tensor_hash([package['noise']])==s['initial_noise_sha256']==s['final_noise_sha256']
  assert package['noise'].shape==(120,128) and not package['noise'].requires_grad
  ds=GeneratorDataset(type('Classes',(),{'num_classes':120})(),cfg,device='cpu');ds.load_from_dict(package)
  assert tensor_hash(ds.generator.parameters())==s['final_generator_sha256']
  with torch.no_grad():difference=float((ds.get_data()[0]-images).abs().max())
  assert s['final_images_match_generator']
 else:
  from synsets.pixels import PixelDataset
  from synsets.base import BaseDistilledDataset
  assert s['adam_step']==5001 and s['initial_parameter_sha256']!=s['final_parameter_sha256']
  ds=PixelDataset.__new__(PixelDataset);BaseDistilledDataset.__init__(ds,device='cpu')
  ds.cfg=cfg;ds.syn_images=package['images'];ds.syn_labels=labels
  regenerated,_=ds.get_data();torch.testing.assert_close(regenerated,images,rtol=1e-5,atol=1e-6)
 assert len(list((ROOT/'checkpoints').glob('step_*.pth')))==11
 assert len(list((ROOT/'images').glob('class_*.png')))==120
 for p in (ROOT/'images').glob('class_*.png'):
  with Image.open(p) as im:assert im.size==(256,256) and im.mode=='RGB';im.verify()
 with (ROOT/'loss.csv').open() as f:rows=list(csv.DictReader(f))
 assert len(rows)==5001 and [int(r['step']) for r in rows]==list(range(5001))
 assert all(math.isfinite(float(r['loss'])) for r in rows)
 for filename,expected in PLAN['preserved_artifacts'].items():assert sha(filename)==expected,filename
 v=dict(status='passed',mode=PLAN['mode'],updates=5001,images_shape=list(images.shape),png_count=120,snapshot_count=11,data_sha256=s['data_sha256'],cpu_regeneration_max_abs_difference=difference,previous_artifacts_unchanged=True)
 with (ROOT/'distillation-validation.json').open('x') as f:json.dump(v,f,indent=2)
 print(json.dumps(v))
if __name__=='__main__':main()
