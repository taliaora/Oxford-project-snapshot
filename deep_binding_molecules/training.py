#import the libraries
import os
import sys
import traceback
from collections import defaultdict
from datetime import datetime
import argparse
import concurrent.futures

import faulthandler

from commons.logger import Logger
from datasets.samplers import HardSampler
from trainer.binding_trainer import BindingTrainer
from datasets.pdbbind import PDBBind
from commons.utils import seed_all, get_random_indices, log
import yaml
from datasets.custom_collate import *
from models import *
from torch.nn import *
from torch.optim import *
from commons.losses import *
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader, Subset
from trainer.metrics import Rsquared, MeanPredictorLoss, MAE, PearsonR, RMSD, RMSDfraction, CentroidDist, \
    CentroidDistFraction, RMSDmedian, CentroidDistMedian, KabschRMSD
from trainer.trainer import Trainer

faulthandler.enable()

def parse_arguments():
    p = argparse.ArgumentParser()

    # General settings
    p.add_argument('--config', type=argparse.FileType(mode='r'), default='configs_clean/RDKitCoords_flexible_self_docking.yml')
    p.add_argument('--experiment_name', type=str, help='name that will be added to the runs folder output')
    p.add_argument('--logdir', type=str, default='runs', help='tensorboard logdirectory')

    # Training parameters
    p.add_argument('--num_epochs', type=int, default=100, help='number of times to iterate through all samples')
    p.add_argument('--batch_size', type=int, default=256, help='samples that will be processed in parallel')
    p.add_argument('--patience', type=int, default=10, help='stop training after no improvement in this many epochs')
    p.add_argument('--minimum_epochs', type=int, default=0, help='minimum number of epochs to run')

    # Dataset parameters
    p.add_argument('--dataset_params', type=dict, default={'param1': 'value1', 'param2': 'value2'}, help='parameters with keywords of the dataset')
    p.add_argument('--dataset', type=str, default='pdbbind', help='which dataset to use')

    # Model and optimization parameters
    p.add_argument('--num_train', type=int, default=5000, help='n samples of the model samples to use for train')
    p.add_argument('--num_val', type=int, default=1000, help='n samples of the model samples to use for validation')
    p.add_argument('--seed', type=int, default=42, help='seed for reproducibility')
    p.add_argument('--multithreaded_seeds', type=list, default=[], help='if this is non empty, multiple threads will be started, training the same model but with the different seeds')
    p.add_argument('--seed_data', type=int, default=42, help='if you want to use a different seed for the datasplit')
    p.add_argument('--loss_func', type=str, default='MSELoss', help='Class name of torch.nn like [MSELoss, L1Loss]')
    p.add_argument('--loss_params', type=dict, default={'param1': 'value1', 'param2': 'value2'}, help='parameters with keywords of the chosen loss function')
    p.add_argument('--optimizer', type=str, default='Adam', help='Class name of torch.optim like [Adam, SGD, AdamW]')
    p.add_argument('--optimizer_params', type=dict, default={'lr': 0.001}, help='parameters with keywords of the chosen optimizer like lr')
    p.add_argument('--clip_grad', type=float, default=5.0, help='clip gradients if magnitude is greater')
    p.add_argument('--lr_scheduler', type=str, default='CosineAnnealingLR', help='Class name of torch.optim.lr_scheduler like [CosineAnnealingLR, ExponentialLR, LambdaLR]')
    p.add_argument('--lr_scheduler_params', type=dict, default={'T_max': 10}, help='parameters with keywords of the chosen lr_scheduler')
    p.add_argument('--scheduler_step_per_batch', default=True, type=bool, help='step every batch if true step every epoch otherwise')
    
    # Logging and Evaluation
    p.add_argument('--log_iterations', type=int, default=-1, help='log every log_iterations iterations (-1 for only logging after each epoch)')
    p.add_argument('--expensive_log_iterations', type=int, default=100, help='frequency with which to do expensive logging operations')
    p.add_argument('--eval_per_epochs', type=int, default=0, help='frequency with which to do run the function run_eval_per_epoch that can do some expensive calculations on the val set or sth like that. If this is zero, then the function will never be called')
    p.add_argument('--metrics', default=['mae'], help='tensorboard metrics [mae, mae_denormalized, qm9_properties ...]')
    p.add_argument('--main_metric', default='loss', help='for early stopping etc.')
    p.add_argument('--main_metric_goal', type=str, default='min', help='controls early stopping. [max, min]')
    p.add_argument('--val_per_batch', type=bool, default=True, help='run evaluation every batch and then average over the eval results. When running the molhiv benchmark for example, this needs to be Fale because we need to evaluate on all val data at once since the metric is rocauc')
    p.add_argument('--tensorboard_functions', default=['function1', 'function2'], help='choices of the TENSORBOARD_FUNCTIONS in utils')
    p.add_argument('--checkpoint', type=str, help='path to directory that contains a checkpoint to continue training')

    # Model and Data Loading
    p.add_argument('--collate_function', default='graph_collate', help='the collate function to use for DataLoader')
    p.add_argument('--collate_params', type=dict, default={'param1': 'value1', 'param2': 'value2'}, help='parameters with keywords of the chosen collate function')
    p.add_argument('--device', type=str, default='cuda', help='What device to train on: cuda or cpu')

    # Model Saving and Loading
    p.add_argument('--models_to_save', type=list, default=[5, 10, 15], help='specify after which epochs to remember the best model')

    # Model Architecture
    p.add_argument('--model_type', type=str, default='MPNN', help='Classname of one of the models in the models dir')
    p.add_argument('--model_parameters', type=dict, default={'param1': 'value1', 'param2': 'value2'}, help='dictionary of model parameters')

    # Training Process
    p.add_argument('--trainer', type=str, default='binding', help='')
    p.add_argument('--train_sampler', type=str, default=None, help='any of pytorchs samplers or a custom sampler')
    p.add_argument('--train_predictions_name', type=str, default='train_preds', help='')
    p.add_argument('--val_predictions_name', type=str, default='val_preds', help='')
    p.add_argument('--sampler_parameters', type=dict, default={'param1': 'value1', 'param2': 'value2'}, help='dictionary of sampler parameters')

    # Evaluation on Test Set
    p.add_argument('--eval_on_test', type=bool, default=True, help='runs evaluation on test set if true')

    # Dataloader Optimization
    p.add_argument('--pin_memory', type=bool, default=True, help='pin memory argument for pytorch dataloaders')
    p.add_argument('--num_workers', type=bool, default=4, help='num workers argument of dataloaders')

    return p.parse_args()


def train_wrap(args):
    # Extracting model and loss parameters from the provided arguments
    mp = args.model_parameters
    lp = args.loss_params

    # Checking if a checkpoint is provided, if yes, use its directory as run_dir
    if args.checkpoint:
        run_directory = os.path.dirname(args.checkpoint)
    else:
        # Creating run_dir based on experiment configuration
        if args.trainer == 'torsion':
            run_directory = f'{args.logdir}/{os.path.splitext(os.path.basename(args.config))[0]}_{args.experiment_name}_layers{mp["n_lays"]}_bs{args.batch_size}_dim{mp["iegmn_lay_hid_dim"]}_nAttH{mp["num_att_heads"]}_norm{mp["layer_norm"]}_normc{mp["layer_norm_coords"]}_normf{mp["final_h_layer_norm"]}_recAtoms{mp["use_rec_atoms"]}_numtrain{args.num_train}_{start_time}'
        else:
            run_directory = f'{args.logdir}/{os.path.splitext(os.path.basename(args.config))[0]}_{args.experiment_name}_layers{mp["n_lays"]}_bs{args.batch_size}_otL{lp["ot_loss_weight"]}_iL{lp["intersection_loss_weight"]}_dim{mp["iegmn_lay_hid_dim"]}_nAttH{mp["num_att_heads"]}_norm{mp["layer_norm"]}_normc{mp["layer_norm_coords"]}_normf{mp["final_h_layer_norm"]}_recAtoms{mp["use_rec_atoms"]}_numtrain{args.num_train}_{start_time}'
    
    # Creating the directory if it does not exist
    if not os.path.exists(run_directory):
        os.mkdir(run_directory)

    # Redirecting stdout and stderr to log files in the run_dir
    sys.stdout = Logger(logpath=os.path.join(run_dir, f'log.log'), syspart=sys.stdout)
    sys.stderr = Logger(logpath=os.path.join(run_dir, f'log.log'), syspart=sys.stderr)

    # Calling the train function with the provided arguments and run_dir
    return train(args, run_directory)


def get_trainer(args, model, data, device, metrics, run_dir, sampler=None):
    # Selecting the appropriate trainer class based on the specified trainer type in args
    if args.trainer is None:
        trainer = Trainer  # Default trainer if not specified
    elif args.trainer == 'binding':
        trainer = BindingTrainer

    # Creating and returning an instance of the selected trainer
    return trainer(model=model, args=args, metrics=metrics, main_metric=args.main_metric,
                   main_metric_goal=args.main_metric_goal, optim=globals()[args.optimizer],
                   loss_func=globals()[args.loss_func](**args.loss_params), device=device, scheduler_step_per_batch=args.scheduler_step_per_batch,
                   run_dir=run_directory, sampler=sampler)


def load_model(args, data_sample, device, **kwargs):
    # Dynamically loading the specified model type using globals()
    model = globals()[args.model_type](device=device,
                                       lig_input_edge_feats_dim=data_sample[0].edata['feat'].shape[1],
                                       rec_input_edge_feats_dim=data_sample[1].edata['feat'].shape[1],
                                       **args.model_parameters, **kwargs)
    return model



def train(args, run_dir):
    # Seed for reproducibility
    seed_all(args.seed)

    # Determine the device for training based on GPU availability
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == 'cuda' else "cpu")

    # Define a dictionary of metrics to be used during training and evaluation
    metrics_dict = {
        'rsquared': Rsquared(),
        'mean_rmsd': RMSD(),
        'rmsd_less_than_20': RMSDfraction(20),
        'rmsd_less_than_50': RMSDfraction(50),
        'median_rmsd': RMSDmedian(),
        'median_centroid_distance': CentroidDistMedian(),
        'centroid_distance_less_than_2': CentroidDistFraction(2),
        'centroid_distance_less_than_5': CentroidDistFraction(5),
        'centroid_distance_less_than_10': CentroidDistFraction(10),
        'centroid_distance_less_than_20': CentroidDistFraction(20),
        'centroid_distance_less_than_50': CentroidDistFraction(50),
        'kabsch_rmsd': KabschRMSD(),
        # Add other metrics or arguments as needed
    }

    # Load training and validation datasets
    train_data = PDBBind(device=device, complex_names_path=args.train_names,
                        lig_predictions_name=args.train_predictions_name, is_train_data=True, **args.dataset_params)
    val_data = PDBBind(device=device, complex_names_path=args.val_names,
                      lig_predictions_name=args.val_predictions_name, **args.dataset_params)

    # Subset the datasets if specified number of samples is provided
    if args.num_train is not None:
        train_data = Subset(train_data, get_random_indices(len(train_data))[:args.num_train])
    if args.num_val is not None:
        val_data = Subset(val_data, get_random_indices(len(val_data))[:args.num_val])

    # Log the sizes of the training and validation datasets
    log('train size: ', len(train_data))
    log('val size: ', len(val_data))

    # Load the model
    model = load_model(args, data_sample=train_data[0], device=device)
    
    # Log the number of trainable parameters in the model
    log('trainable params in model: ', sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Determine the collate function based on arguments
    collate_function = globals()[args.collate_function] if args.collate_params == {} else globals()[args.collate_function](**args.collate_params)

    # Create data loaders for training and validation
    if args.train_sampler is not None:
        # Use a custom sampler if specified
        sampler = globals()[args.train_sampler](data_source=train_data, batch_size=args.batch_size)
        train_loader = DataLoader(train_data, batch_sampler=sampler, collate_fn=collate_function,
                                  pin_memory=args.pin_memory, num_workers=args.num_workers)
    else:
        # Use default shuffling if no custom sampler is specified
        sampler = None
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_function,
                                  pin_memory=args.pin_memory, num_workers=args.num_workers)
    
    # Create a validation data loader
    val_loader = DataLoader(val_data, batch_size=args.batch_size, collate_fn=collate_function,
                            pin_memory=args.pin_memory, num_workers=args.num_workers)

    # Select metrics for training
    metrics = {metric: metrics_dict[metric] for metric in args.metrics}

    # Create the trainer instance
    trainer = get_trainer(args=args, model=model, data=train_data, device=device, metrics=metrics, run_dir=run_dir,
                          sampler=sampler)

    # Train the model and obtain validation metrics
    val_metrics, _, _ = trainer.train(train_loader, val_loader)

    # If evaluation on the test set is enabled
    if args.eval_on_test:
        # Load the test dataset
        test_data = PDBBind(device=device, complex_names_path=args.test_names, **args.dataset_params)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, collate_fn=collate_function,
                                 pin_memory=args.pin_memory, num_workers=args.num_workers)
        log('test size: ', len(test_data))
        
        # Evaluate on the test set and return validation and test metrics along with log directory
        test_metrics, _, _ = trainer.evaluation(test_loader, data_split='test')
        return val_metrics, test_metrics, trainer.writer.log_dir

    # Return validation metrics if test evaluation is not required
    return val_metrics






def get_arguments():
    # Parse command-line arguments
    args = parse_arguments()

    # If a configuration file is provided
    if args.config:
        # Load configuration settings from the file
        config_dict = yaml.load(args.config, Loader=yaml.FullLoader)
        arg_dict = args.__dict__

        # Update arguments with values from the configuration file
        for key, value in config_dict.items():
            if isinstance(value, list):
                for v in value:
                    arg_dict[key].append(v)
            else:
                arg_dict[key] = value

        # Update the 'config' attribute to store the configuration file name
        args.config = args.config.name
    else:
        config_dict = {}

    # If a checkpoint file is provided
    if args.checkpoint:
        # Load arguments from the checkpoint file
        arg_dict = args.__dict__
        with open(os.path.join(os.path.dirname(args.checkpoint), 'train_arguments.yaml'), 'r') as arg_file:
            checkpoint_dict = yaml.load(arg_file, Loader=yaml.FullLoader)

        # Update arguments with values from the checkpoint file, avoiding conflicts with config file
        for key, value in checkpoint_dict.items():
            if key not in config_dict.keys():
                if isinstance(value, list):
                    for v in value:
                        arg_dict[key].append(v)
                else:
                    arg_dict[key] = value

    return args

