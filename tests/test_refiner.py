"""CPU correctness, frozen-backbone and exact continuation checks."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
import random
import numpy as np
import torch
from models import GATraj
from refinement import GATrajRefiner, ResidualRefiner, file_sha256
from Processor import Processor
from train import get_parser, experiment_name, prepare_seed, parse_run_args
from compare import compare


def args(**changes):
    a = get_parser().parse_args([])
    for key, value in changes.items(): setattr(a, key, value)
    return a


def example():
    absolute = torch.arange(20*3*2, dtype=torch.float32).reshape(20,3,2)/100
    normalized = absolute - absolute[7:8]
    mask = np.ones((3,3))-np.eye(3)
    return absolute, normalized, [np.repeat(mask[None],20,0)], torch.full((3,20),2.), np.array([[0,3]])


class RefinerTests(unittest.TestCase):
    def setUp(self):
        root=Path(__file__).resolve().parents[1]/'outputs'; root.mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=root); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        prepare_seed(7)
        with contextlib.redirect_stdout(io.StringIO()): self.base=GATraj(args(stage='backbone')).eval()
        self.path=self.root/'original.tar'
        config=vars(args(stage='backbone')).copy()
        torch.save({'epoch':1000,'state_dict':self.base.state_dict(),'experiment_config':config},self.path)
        self.digest=file_sha256(self.path)

    def wrapper(self, **changes):
        options=dict(base_checkpoint=str(self.path),base_sha256=file_sha256(self.path))
        options.update(changes)
        with contextlib.redirect_stdout(io.StringIO()): return GATrajRefiner(args(**options))

    def test_identity_and_bypass_with_same_backbone(self):
        m=self.wrapper().eval(); x=example()
        self.base.requires_grad_(False)  # Match the frozen inference kernel path.
        with torch.no_grad():
            original=self.base(x,1000,return_mixture=True)
            refined=m(x,1,return_mixture=True)
            for a,b in zip(original,refined): torch.testing.assert_close(a,b,rtol=0,atol=0)
            torch.nn.init.normal_(m.refinement.network[-1].weight,0,.03)
            m.args.refiner=False
            for a,b in zip(original,m(x,1,return_mixture=True)):
                torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_frozen_backbone_gradients_and_training_step(self):
        m=self.wrapper().train(); x=example()
        self.assertFalse(m.backbone.training)
        before={k:v.clone() for k,v in m.backbone.state_dict().items()}
        optimizer=torch.optim.Adam((p for p in m.parameters() if p.requires_grad),lr=1e-4)
        loss,_=m(x,1); loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is None for p in m.backbone.parameters()))
        self.assertGreater(m.refinement.network[-1].weight.grad.abs().sum().item(),0)
        optimizer.step()
        for key,value in before.items(): torch.testing.assert_close(value,m.backbone.state_dict()[key],rtol=0,atol=0)
        self.assertGreater(m.refinement(x[1][:8],m.backbone(x,1,return_mixture=True)[0],m.backbone.hidden_state_global).abs().sum().item(),0)

    def test_endpoint_shape_separation_and_bounds(self):
        r=ResidualRefiner(args())
        torch.nn.init.normal_(r.network[-1].weight,0,.1)
        x=example(); means=torch.randn(20,3,12,2); context=torch.randn(3,64)
        self.assertTrue((r.basis[[0,-1]]==0).all())
        self.assertEqual(r.ramp[0].item(),0)
        r.args.shape_refine=False; endpoint=r(x[1][:8],means,context)
        torch.testing.assert_close(endpoint, r.ramp[1:][None,None,:,None]*endpoint[:,:,-1:])
        self.assertLessEqual(endpoint.abs().max().item(),r.args.max_endpoint_correction)
        r.args.shape_refine=True;r.args.endpoint_refine=False
        shape=r(x[1][:8],means,context)
        self.assertTrue((shape[:,:,-1]==0).all())
        self.assertLessEqual(shape.abs().max().item(),r.args.max_shape_correction+1e-6)

    def test_no_future_label_leakage(self):
        m=self.wrapper().eval()
        torch.nn.init.normal_(m.refinement.network[-1].weight,0,.03)
        x=example(); changed=list(x)
        changed[0]=x[0].clone(); changed[1]=x[1].clone()
        changed[0][8:]+=100; changed[1][8:]-=100
        with torch.no_grad():
            for a,b in zip(m(x,1,return_mixture=True),m(tuple(changed),1,return_mixture=True)):
                torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_reject_wrong_fold_and_changed_source(self):
        with self.assertRaisesRegex(ValueError,'test fold'): self.wrapper(test_set=0)
        with self.assertRaisesRegex(ValueError,'SHA256'): self.wrapper(base_sha256='wrong')
        with self.assertRaisesRegex(ValueError,'validation split'): self.wrapper(validation_set=4)

    def test_checkpoint_name_allows_eval_bypass_and_separates_components(self):
        a=args(base_sha256='a');b=args(base_sha256='a',refiner=False)
        self.assertEqual(experiment_name(a),experiment_name(b))
        self.assertNotEqual(experiment_name(a),experiment_name(args(base_sha256='b')))
        self.assertNotEqual(experiment_name(a),experiment_name(args(base_sha256='a',shape_refine=False)))

    def test_validation_only_selection_and_cli_resolution(self):
        def report(variant, split, ade, fde):
            return {'variant':variant,'split':split,'test_set':1,'seed':1,'epoch':100,
                    'base_epoch':1000,'base_sha256':self.digest,'evaluation_scene':1,'num_modes':20,
                    'groups':{'all':{'agents':10,'minADE':ade,'minFDE':fde}},
                    'experiment':{'train_model':'run','num_epochs':100}}
        a=report('baseline','validation',.1,.15)
        b=report('refined','validation',.09,.16)
        self.assertEqual(compare(a,b)['validation_choice'],'baseline')
        b['groups']['all']['minFDE']=.14
        self.assertEqual(compare(a,b)['validation_choice'],'refined')
        a['split']=b['split']='test'
        self.assertNotIn('validation_choice',compare(a,b))
        flags=['--stage','refiner','--base_checkpoint',str(self.path),'--base_sha256',self.digest,
               '--using_cuda','False','--save_base_dir',str(self.root),'--num_epochs','100']
        expected=experiment_name(get_parser().parse_args(flags))
        parsed=parse_run_args(argv=flags)
        self.assertEqual(expected,parsed.train_model)
        evaluated=parse_run_args(default_phase='test',argv=[*flags,'--phase','test','--refiner','False','--load_model','100'])
        self.assertEqual(parsed.train_model,evaluated.train_model)

    def processor(self, load=0):
        prepare_seed(9)
        p=Processor.__new__(Processor)
        p.args=args(base_checkpoint=str(self.path),base_sha256=self.digest,
                    stage='refiner',save_dir=str(self.root),model_dir=str(self.root/'run'),
                    train_model='run',load_model=load,num_epochs=100,learning_rate=1e-4)
        with contextlib.redirect_stdout(io.StringIO()): p.net=GATrajRefiner(p.args)
        p.lr=p.args.learning_rate;p.set_optimizer();p.epoch=1;p.load_model()
        return p

    def step(self,p,epoch):
        x=list(example()); noise=torch.randn_like(x[0])*.001 + np.random.random()*.001 + random.random()*.001
        x[0]=x[0]+noise; x[1]=x[0]-x[0][7:8]
        p.net.train();p.optimizer.zero_grad();p.net(tuple(x),epoch)[0].backward()
        p.optimizer.step();p.scheduler.step()

    def test_exact_full_wrapper_resume_and_self_contained_checkpoint(self):
        (self.root/'run').mkdir()
        full=self.processor()
        for epoch in range(1,5): self.step(full,epoch)
        segmented=self.processor()
        for epoch in range(1,3): self.step(segmented,epoch)
        segmented.save_model(2)
        self.path.unlink()  # Resume restores the embedded backbone without reading its source file.
        resumed=self.processor(load=2)
        self.assertEqual(resumed.epoch,3)
        self.assertEqual(resumed.net.base_epoch,1000)
        for epoch in range(3,5): self.step(resumed,epoch)
        for key,value in full.net.state_dict().items():
            torch.testing.assert_close(value,resumed.net.state_dict()[key],rtol=0,atol=0)
        self.assertEqual(full.scheduler.state_dict(),resumed.scheduler.state_dict())


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
