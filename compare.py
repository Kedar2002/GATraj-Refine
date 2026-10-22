"""Paired bypass/refined reports from the same frozen original checkpoint."""
import argparse
import json
from pathlib import Path


def compare(baseline, refined):
    if baseline['variant'] != 'baseline' or refined['variant'] != 'refined':
        raise ValueError('Expected original bypass and refined reports')
    for key in ('test_set','seed','epoch','split','evaluation_scene','base_sha256','base_epoch','num_modes'):
        if baseline[key] != refined[key]: raise ValueError('Unmatched reports: '+key)
    if baseline['experiment']['train_model'] != refined['experiment']['train_model']:
        raise ValueError('Both reports must come from the same refiner checkpoint')
    if baseline['epoch'] != baseline['experiment']['num_epochs']:
        raise ValueError('Use the predetermined final refiner epoch for comparison')
    a,b=baseline['groups']['all'],refined['groups']['all']
    if a['agents']!=b['agents']: raise ValueError('Agent-window counts differ')
    metrics=('minADE','minFDE')
    if any(a[k] is None or b[k] is None for k in metrics): raise ValueError('Empty evaluation split')
    result={'split':baseline['split'],'test_set':baseline['test_set'],'seed':baseline['seed'],
            'base_sha256':baseline['base_sha256'],
            'baseline':{k:a[k] for k in metrics},'refined':{k:b[k] for k in metrics},
            'delta_refined_minus_baseline':{k:b[k]-a[k] for k in metrics}}
    if baseline['split']=='validation':
        improvement=all(b[k]<=a[k] for k in metrics) and any(b[k]<a[k] for k in metrics)
        result['validation_choice']='refined' if improvement else 'baseline'
        result['note']='Validation choice only. It does not guarantee better unseen-test results.'
    else:
        result['note']='Test reporting only; no checkpoint or method selected from test scores.'
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('baseline',type=Path);p.add_argument('refined',type=Path)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=compare(json.loads(a.baseline.read_text()),json.loads(a.refined.read_text()))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
