"""Read one completed job's accounting without any submission."""
import argparse
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('phase',choices=['distill','eval'])
parser.add_argument('seed',type=int,choices=[42,123,2026])
args=parser.parse_args()
run=ROOT/f'seed_{args.seed}'/args.phase
job=json.loads((run/'submission.json').read_text())['job_id']
output=subprocess.check_output(['qacct','-j',job],text=True)
fields={}
for line in output.splitlines():
    parts=line.split(None,1)
    if len(parts)==2: fields[parts[0]]=parts[1].strip()
assert fields['jobnumber']==job
keys=['jobnumber','hostname','group','priority','start_time','end_time','exit_status','failed','wallclock','ru_wallclock','slots','maxrss']
with (run/'job-accounting.json').open('x') as handle: json.dump({k:fields[k] for k in keys if k in fields},handle,indent=2)
with (run/'logs/qacct.txt').open('x') as handle: handle.write(output)
print(fields['exit_status'])
