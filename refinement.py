"""Endpoint and smooth path residuals around an unchanged, frozen GATraj."""
import hashlib
import math
import warnings
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from models import GATraj


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class ResidualRefiner(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.parts = args.shape_components
        # Include t=0 explicitly to verify the observation boundary.
        u = torch.linspace(0, 1, args.pred_length + 1)
        degree = self.parts + 1
        basis = torch.stack([math.comb(degree, j) * u.pow(j) * (1-u).pow(degree-j)
                             for j in range(1, degree)], dim=1)
        self.register_buffer('ramp', u)
        self.register_buffer('basis', basis)
        features = 2*(args.obs_length-1) + 2*args.pred_length + args.hidden_size
        self.network = nn.Sequential(nn.Linear(features, args.refiner_hidden), nn.LayerNorm(args.refiner_hidden),
                                     nn.SiLU(), nn.Linear(args.refiner_hidden, 2*(1+self.parts)))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, observed, means, context):
        # means [K,N,T,2], observed [O,N,2], context [N,D].
        k, n, t, _ = means.shape
        velocity = (observed[1:] - observed[:-1]).permute(1, 0, 2).reshape(n, -1)
        features = torch.cat((velocity[None].expand(k, -1, -1), means.flatten(2),
                              context[None].expand(k, -1, -1)), dim=-1)
        values = self.network(features).reshape(k, n, 1+self.parts, 2)
        goal = self.args.max_endpoint_correction * torch.tanh(values[:, :, 0] / self.args.max_endpoint_correction)
        controls = self.args.max_shape_correction * torch.tanh(values[:, :, 1:] / self.args.max_shape_correction)
        correction = torch.zeros_like(means)
        if self.args.endpoint_refine:
            correction = correction + self.ramp[1:][None, None, :, None] * goal[:, :, None]
        if self.args.shape_refine:
            correction = correction + torch.einsum('tj,knjc->kntc', self.basis[1:], controls)
        return correction


class GATrajRefiner(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.backbone = GATraj(args)
        self.refinement = ResidualRefiner(args)
        self.base_epoch = None
        if args.load_model == 0:
            self.load_base(args.base_checkpoint)
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    def load_base(self, path):
        if not path or not Path(path).is_file():
            raise FileNotFoundError('Set --base_checkpoint to a trusted original GATraj checkpoint')
        actual = file_sha256(path)
        if self.args.base_sha256 and actual != self.args.base_sha256:
            raise ValueError('Base checkpoint SHA256 differs from the configured checkpoint')
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        config = checkpoint.get('experiment_config', {})
        if config.get('awta') or config.get('motion_module') or config.get('fix_attention'):
            raise ValueError('Use an original GATraj baseline, not aWTA, RiskMotion or corrected-attention weights')
        if config and config.get('test_set', self.args.test_set) != self.args.test_set:
            raise ValueError('Base checkpoint belongs to another test fold')
        validation = config.get('validation_set', -1)
        if validation != self.args.validation_set:
            raise ValueError('Base checkpoint validation split differs; validation must be excluded from backbone training')
        for key in ('obs_length', 'pred_length', 'final_mode', 'hidden_size', 'x_encoder_layers',
                    'x_encoder_head', 'input_offset', 'input_position', 'input_mix', 'SR',
                    'pass_time', 'mlp_decoder', 'ifGaussian', 'neighbor_thred'):
            if key in config and config[key] != getattr(self.args, key):
                raise ValueError('Base checkpoint setting differs: ' + key)
        if not config:
            warnings.warn('Legacy base has no experiment metadata; confirm its fold, architecture and training split manually.')
        self.backbone.load_state_dict(checkpoint['state_dict'], strict=True)
        self.base_epoch = int(checkpoint.get('epoch', 0))
        self.args.base_sha256 = actual
        print('Loaded frozen original GATraj:', path, 'epoch:', self.base_epoch, 'SHA256:', actual)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()  # Frozen dropout must stay disabled during refiner training.
        return self

    def forward(self, inputs, epoch, iftest=False, return_mixture=False):
        with torch.no_grad():
            means, scales, logits = self.backbone(inputs, epoch, iftest=True, return_mixture=True)
        if self.args.refiner:
            correction = self.refinement(inputs[1][:self.args.obs_length, :, :2], means,
                                         self.backbone.hidden_state_global.detach())
            refined = means + correction
        else:
            correction = torch.zeros_like(means)
            refined = means
        mixture = refined, scales, logits
        if return_mixture:
            return mixture  # Labels never enter prediction/refinement.
        target = inputs[1][self.args.obs_length:, :, :2]
        loss, forecasts = self.backbone.mdn_loss(target, mixture, 1, iftest)
        if self.training and not iftest:
            y = target.transpose(0, 1)
            # Keep the original backbone's mode assignment; do not collapse every mode toward one future.
            winner = (means-y).norm(dim=-1).sum(dim=-1).argmin(dim=0)
            agents = torch.arange(y.shape[0], device=y.device)
            chosen = torch.cat((refined, scales), dim=-1)[winner, agents]
            regression = self.backbone.reg_loss(chosen, y)
            distances = (means-y).norm(dim=-1).mean(dim=-1)
            classification = self.backbone.cls_loss(logits, torch.softmax(-distances, dim=0).t().detach())
            endpoint = F.smooth_l1_loss(refined[winner, agents, -1], y[:, -1])
            penalty = correction.square().mean()
            loss = regression + classification + self.args.endpoint_loss_weight*endpoint + self.args.correction_penalty*penalty
        return loss, forecasts
