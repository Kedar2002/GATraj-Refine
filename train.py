'''
Train script
Author: Mengmeng Liu
Date: 2022/09/24
'''
import argparse
import ast
from Processor import *
import numpy as np
import random
import torch
import yaml

def get_parser():
    parser = argparse.ArgumentParser(description='GATraj')
    parser.add_argument(
        '--ifvalid',default=True,type=ast.literal_eval,
        help="=False,use all train set to train,"
             "=True,use train set to train and valid")
    parser.add_argument(
        '--mlp_decoder',default=False,type=ast.literal_eval)
    parser.add_argument(
        '--input_offset',default=True,type=ast.literal_eval)
    parser.add_argument(
        '--input_position',default=False,type=ast.literal_eval)
    parser.add_argument(
        '--input_mix',default=False,type=ast.literal_eval)
    parser.add_argument(
        '--ifGaussian',default=False,type=ast.literal_eval)
    parser.add_argument(
        '--SR',default=True,type=ast.literal_eval)
    parser.add_argument("--pass_time",
                        default=2,
                        type=int)
    parser.add_argument("--final_mode",
                        default=20,
                        type=int)
    parser.add_argument("--T_max",
                        default=1000,
                        type=int)
    parser.add_argument("--eta_min",
                        default=3e-6,
                        type=float)
    parser.add_argument(
        '--output_size',default=2,type=int)
    parser.add_argument(
        '--input_size',default=2,type=int)
    parser.add_argument(
        '--min_obs',default=8,type=int)
    parser.add_argument('--num_pred', type=int, default=1, help='This is the number of predictions for each agent')
    parser.add_argument('--ratio', type=float, default=0.95, help='The overlap ratio of coexisting for group detection')
    parser.add_argument('--z_dim', type=int, default=32, help='This is the size of the latent variable')
    parser.add_argument('--hidden_size', type=int, default=64, help='The size of LSTM hidden state')
    parser.add_argument('--x_encoder_layers', type=int, default=3, help='Number of transformer block layers for x_encoder')
    parser.add_argument('--x_encoder_head', type=int, default=8, help='Head number of x_encoder')
    parser.add_argument(
        '--gpu', default=0,type=int,
        help='gpu id')
    parser.add_argument(
        '--using_cuda',default=True,type=ast.literal_eval) # We did not test on cpu
    # You may change these arguments (model selection and dirs)
    parser.add_argument(
        '--test_set',default=1,type=int,
        help='Set this value to 0~4 for ETH-univ, ETH-hotel, UCY-zara01, UCY-zara02, UCY-univ')
    parser.add_argument(
        '--base_dir',default='.',
        help='Base directory including these scrits.')
    parser.add_argument(
        '--save_base_dir',default='./savedata/',
        help='Directory for saving caches and models.')
    parser.add_argument(
        '--phase', default='train',
        help='Set this value to \'train\' or \'test\'')
    parser.add_argument(
        '--GT',default=True)
    parser.add_argument(
        '--train_model', default=None,
        help='Your model name')
    parser.add_argument(
        '--load_model', default=0,type=int,
        help="load model weights from this index before training or testing")
    parser.add_argument(
        '--epochs_per_run', default=0, type=int,
        help='Optional number of epochs to run before stopping; num_epochs remains the full schedule length')
    parser.add_argument(
        '--seed', default=1, type=int,
        help='Random seed for training and deterministic data ordering')
    parser.add_argument(
        '--model', default=None)
    parser.add_argument('--stage', choices=('backbone', 'refiner'), default='refiner')
    parser.add_argument('--base_checkpoint', default='', help='Trusted original checkpoint for frozen-backbone initialization')
    parser.add_argument('--base_sha256', default='', help='Optional expected fingerprint; required if the source file is absent on resume')
    parser.add_argument('--refiner', default=True, type=ast.literal_eval, help='False bypasses refinement for evaluation')
    parser.add_argument('--endpoint_refine', default=True, type=ast.literal_eval)
    parser.add_argument('--shape_refine', default=True, type=ast.literal_eval)
    parser.add_argument('--shape_components', default=3, type=int)
    parser.add_argument('--refiner_hidden', default=64, type=int)
    parser.add_argument('--max_endpoint_correction', default=1.0, type=float, help='Per-coordinate correction bound in metres')
    parser.add_argument('--max_shape_correction', default=0.5, type=float, help='Per-coordinate shape-control bound in metres')
    parser.add_argument('--endpoint_loss_weight', default=0.1, type=float)
    parser.add_argument('--correction_penalty', default=0.01, type=float)
    parser.add_argument('--validation_set', default=-1, type=int,
                        help='Optional whole validation scene excluded from training; -1 uses the standard test split')
    ######################################
    parser.add_argument(
        '--dataset',default='eth5')
    parser.add_argument(
        '--save_dir')
    parser.add_argument(
        '--model_dir')
    parser.add_argument(
        '--config')
    parser.add_argument(
        '--val_fraction',default=0,type=float)
    #Perprocess
    parser.add_argument(
        '--seq_length',default=20,type=int)
    parser.add_argument(
        '--obs_length',default=8,type=int)
    parser.add_argument(
        '--pred_length',default=12,type=int)
    parser.add_argument(
        '--batch_size',default=64,type=int)
    parser.add_argument(
        '--show_step',default=40,type=int)
    parser.add_argument(
        '--step_ratio',default=0.5,type=int)
    parser.add_argument(
        '--lr_step',default=20,type=int)
    parser.add_argument(
        '--num_epochs',default=1000,type=int)
    parser.add_argument(
        '--ifshow_detail',default=True,type=ast.literal_eval)
    parser.add_argument(
        '--randomRotate',default=True,type=ast.literal_eval,
        help="=True:random rotation of each trajectory fragment")
    parser.add_argument(
        '--neighbor_thred',default=10,type=int)
    parser.add_argument(
        '--learning_rate',default=1e-04,type=float)
    parser.add_argument(
        '--clip',default=10,type=int)
    return parser
def save_arg(args):
    # save arg
    arg_dict = vars(args)
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    with open(args.config, 'w') as f:
        yaml.dump(arg_dict, f)
def prepare_seed(rand_seed):
	# Seed every RNG used by training and data preprocessing.
	np.random.seed(rand_seed)
	random.seed(rand_seed)
	torch.manual_seed(rand_seed)
	torch.cuda.manual_seed_all(rand_seed)
	if hasattr(torch.backends, 'cudnn'):
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False

def experiment_name(args):
    import hashlib, json
    config = vars(args).copy()
    config['model'] = args.model or ('models.GATraj' if args.stage == 'backbone' else 'refinement.GATrajRefiner')
    for key in ('phase', 'load_model', 'epochs_per_run', 'train_model', 'base_checkpoint',
                'save_base_dir', 'save_dir', 'model_dir', 'config', 'gpu', 'using_cuda', 'ifshow_detail', 'show_step'):
        config.pop(key, None)
    if args.stage == 'backbone':
        for key in ('base_sha256', 'refiner', 'endpoint_refine', 'shape_refine', 'shape_components',
                    'refiner_hidden', 'max_endpoint_correction', 'max_shape_correction',
                    'endpoint_loss_weight', 'correction_penalty'):
            config.pop(key, None)
    else:
        config.pop('refiner', None)  # On/off evaluation must address the same checkpoint directory.
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    return f'GATraj_{args.stage}_s{args.seed}_{digest}'


def parse_run_args(default_phase='train', argv=None):
    parser = get_parser()
    parser.set_defaults(phase=default_phase)
    args = parser.parse_args(argv)
    if not 0 <= args.test_set <= 4 or not -1 <= args.validation_set <= 4:
        parser.error('test_set must be 0..4 and validation_set -1..4')
    if args.validation_set == args.test_set:
        parser.error('Validation scene must differ from the held-out test scene')
    if args.num_epochs < 1 or args.epochs_per_run < 0:
        parser.error('num_epochs must be positive and epochs_per_run nonnegative')
    if min(args.shape_components, args.refiner_hidden) < 1 or min(args.max_endpoint_correction, args.max_shape_correction) <= 0:
        parser.error('Refiner dimensions and bounds must be positive')
    if min(args.endpoint_loss_weight, args.correction_penalty) < 0:
        parser.error('Loss weights must be nonnegative')
    if args.stage == 'refiner':
        from refinement import file_sha256
        if args.phase == 'train' and (not args.refiner or not (args.endpoint_refine or args.shape_refine)):
            parser.error('Refiner training requires refinement and at least one correction component')
        if args.base_checkpoint and os.path.isfile(args.base_checkpoint):
            actual = file_sha256(args.base_checkpoint)
            if args.base_sha256 and args.base_sha256 != actual:
                parser.error('Base checkpoint fingerprint changed')
            args.base_sha256 = actual
        elif args.load_model == 0 or not args.base_sha256:
            parser.error('Supply an existing --base_checkpoint; self-contained resume needs --base_sha256')
    args.model = args.model or ('models.GATraj' if args.stage == 'backbone' else 'refinement.GATrajRefiner')
    if args.validation_set >= 0 and args.val_fraction != 0:
        parser.error('Use whole-scene validation or val_fraction, not both')
    if args.train_model is None:
        args.train_model = experiment_name(args)
    args.save_dir = os.path.join(args.save_base_dir, str(args.test_set))
    args.model_dir = os.path.join(args.save_dir, args.train_model)
    args.config = os.path.join(args.model_dir, 'config_' + args.phase + '.yaml')
    # CLI is authoritative; checkpoint metadata validates compatibility on resume.
    prepare_seed(args.seed)
    save_arg(args)
    if args.using_cuda:
        torch.cuda.set_device(args.gpu)
    return args


if __name__ == '__main__':
    args = parse_run_args()
    processor = Processor(args)
    if args.phase == 'test':
        processor.playtest()
    else:
        processor.playtrain()
