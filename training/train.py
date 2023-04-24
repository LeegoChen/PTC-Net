# Code adapted or modified from MinkLoc3DV2 repo: https://github.com/jac99/MinkLoc3Dv2

# Train PTC-Net model
import os
import sys
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)
import argparse
import torch

from training.trainer import do_train
from misc.utils import TrainingParams


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train PTC-Net')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--model_config', type=str, required=True, help='Path to the model-specific configuration file')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--resume', type=str, default=None, help='resume')  # add resume
    parser.set_defaults(debug=False)

    args = parser.parse_args()
    print('Training config path: {}'.format(args.config))
    print('Model config path: {}'.format(args.model_config))
    print('Debug mode: {}'.format(args.debug))

    if args.resume is not None:
        resume_filename = args.resume
        print("Resuming From {}".format(resume_filename))
        saved_state_dict = torch.load(resume_filename)
        params = saved_state_dict['params']
    else:
        params = TrainingParams(args.config, args.model_config, debug=args.debug)
        saved_state_dict = None

    params.print()

    if args.debug:
        torch.autograd.set_detect_anomaly(True)

    do_train(params, saved_state_dict=saved_state_dict)
