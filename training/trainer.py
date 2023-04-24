# Code adapted or modified from PPT_Net repo: https://github.com/jac99/MinkLoc3Dv2

import os
import numpy as np
import torch
import tqdm
import pathlib
# import wandb

from misc.utils import TrainingParams, get_datetime
from models.losses.loss import make_losses

from datasets.dataset_utils import make_dataloaders
from eval.pnv_evaluate import evaluate, pnv_write_eval_stats
from models.ptcnet import PTC_Net, PTC_Net_L



def print_global_stats(phase, stats):
    if 'loss1' not in stats:
        s = f"{phase}  loss: {stats['loss']:.4f}    embedding norm: {stats['avg_embedding_norm']:.3f}   "
    else:
        s = f"{phase}  loss: {stats['loss']:.4f}    loss1: {stats['loss1']:.4f}   loss2: {stats['loss2']:.4f}   "

    if 'num_triplets' in stats:
        s += f"Triplets (all/active): {stats['num_triplets']:.1f}/{stats['num_non_zero_triplets']:.1f}  " \
             f"Mean dist (pos/neg): {stats['mean_pos_pair_dist']:.3f}/{stats['mean_neg_pair_dist']:.3f}   "
    if 'positives_per_query' in stats:
        s += f"#positives per query: {stats['positives_per_query']:.1f}   "
    if 'best_positive_ranking' in stats:
        s += f"best positive rank: {stats['best_positive_ranking']:.1f}   "
    if 'recall' in stats:
        s += f"Recall@1: {stats['recall'][1]:.4f}   "
    if 'ap' in stats:
        s += f"AP: {stats['ap']:.4f}   "

    print(s)


def print_stats(phase, stats):
    print_global_stats(phase, stats['global'])


def tensors_to_numbers(stats):
    stats = {e: stats[e].item() if torch.is_tensor(stats[e]) else stats[e] for e in stats}
    return stats


def training_step(global_iter, model, phase, device, optimizer, loss_fn):
    assert phase in ['train', 'val']

    batch, positives_mask, negatives_mask = next(global_iter)
    batch = {e: batch[e].to(device) for e in batch}

    if phase == 'train':
        model.train()
    else:
        model.eval()

    optimizer.zero_grad()

    with torch.set_grad_enabled(phase == 'train'):
        y = model(batch)
        stats = model.stats.copy() if hasattr(model, 'stats') else {}
        embeddings = y['global']

        loss, temp_stats = loss_fn(embeddings, positives_mask, negatives_mask)

        temp_stats = tensors_to_numbers(temp_stats)
        stats.update(temp_stats)
        if phase == 'train':
            loss.backward()
            optimizer.step()

    torch.cuda.empty_cache()  # Prevent excessive GPU memory consumption by SparseTensors

    return stats


def multistaged_training_step(global_iter, model, phase, device, optimizer, loss_fn):
    # Training step using multistaged backpropagation algorithm as per:
    # "Learning with Average Precision: Training Image Retrieval with a Listwise Loss"
    # This method will break when the model contains Dropout, as the same mini-batch will produce different embeddings.
    # Make sure mini-batches in step 1 and step 3 are the same (so that BatchNorm produces the same results)
    # See some exemplary implementation here: https://gist.github.com/ByungSun12/ad964a08eba6a7d103dab8588c9a3774

    assert phase in ['train', 'val']
    batch, positives_mask, negatives_mask = next(global_iter)

    if phase == 'train':
        model.train()
    else:
        model.eval()

    # Stage 1 - calculate descriptors of each batch element (with gradient turned off)
    # In training phase network is in the train mode to update BatchNorm stats
    embeddings_l = []
    with torch.set_grad_enabled(False):
        for minibatch in batch:
            minibatch = {e: minibatch[e].to(device) for e in minibatch}
            y = model(minibatch)
            embeddings_l.append(y['global'])

    torch.cuda.empty_cache()  # Prevent excessive GPU memory consumption by SparseTensors

    # Stage 2 - compute gradient of the loss w.r.t embeddings
    embeddings = torch.cat(embeddings_l, dim=0)

    with torch.set_grad_enabled(phase == 'train'):
        if phase == 'train':
            embeddings.requires_grad_(True)
        loss, stats = loss_fn(embeddings, positives_mask, negatives_mask)
        stats = tensors_to_numbers(stats)
        if phase == 'train':
            loss.backward()
            embeddings_grad = embeddings.grad

    # Delete intermediary values
    embeddings_l, embeddings, y, loss = None, None, None, None

    # Stage 3 - recompute descriptors with gradient enabled and compute the gradient of the loss w.r.t.
    # network parameters using cached gradient of the loss w.r.t embeddings
    if phase == 'train':
        optimizer.zero_grad()
        i = 0
        with torch.set_grad_enabled(True):
            for minibatch in batch:
                minibatch = {e: minibatch[e].to(device) for e in minibatch}
                y = model(minibatch)
                embeddings = y['global']
                minibatch_size = len(embeddings)
                # Compute gradients of network params w.r.t. the loss using the chain rule (using the
                # gradient of the loss w.r.t. embeddings stored in embeddings_grad)
                # By default gradients are accumulated
                embeddings.backward(gradient=embeddings_grad[i: i+minibatch_size])
                i += minibatch_size

            optimizer.step()

    torch.cuda.empty_cache()  # Prevent excessive GPU memory consumption by SparseTensors

    return stats


def do_train(params: TrainingParams, saved_state_dict=None):
    # Create model class
    s = get_datetime()
    if params.model_params.model == 'PTC-Net':
        model = PTC_Net(params.model_params)
    elif params.model_params.model == 'PTC-Net-L':
        model = PTC_Net_L(params.model_params)
    else:
        print('Unknown network: {}'.format(params.model_params.model))
        raise NotImplementedError

    if saved_state_dict is not None:
        model.load_state_dict(saved_state_dict['net'])

    model_name = 'model_' + params.model_params.model + '_' + s

    print('Model name: {}'.format(model_name))
    weights_path = create_weights_folder()

    model_pathname = os.path.join(weights_path, model_name)
    if hasattr(model, 'print_info'):
        model.print_info()
    else:
        n_params = sum([param.nelement() for param in model.parameters()])
        print('Number of model parameters: {}'.format(n_params))

    # Move the model to the proper device before configuring the optimizer
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    model.to(device)
    print('Model device: {}'.format(device))

    # set up dataloaders
    # dataloaders = make_dataloaders(params)
    dataloaders = make_dataloaders(params, validation=False)

    loss_fn = make_losses(params)

    # Training elements
    if params.optimizer == 'Adam':
        optimizer_fn = torch.optim.Adam
    elif params.optimizer == 'AdamW':
        optimizer_fn = torch.optim.AdamW
    else:
        raise NotImplementedError(f"Unsupported optimizer: {params.optimizer}")

    if params.weight_decay is None or params.weight_decay == 0:
        optimizer = optimizer_fn(model.parameters(), lr=params.lr)
    else:
        optimizer = optimizer_fn(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)

    if params.scheduler is None:
        scheduler = None
    else:
        if params.scheduler == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params.epochs+1,
                                                                   eta_min=params.min_lr)
        elif params.scheduler == 'MultiStepLR':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, params.scheduler_milestones, gamma=0.1)
        else:
            raise NotImplementedError('Unsupported LR scheduler: {}'.format(params.scheduler))

    if saved_state_dict is None:
        start_epoch = 0
    else:
        start_epoch = saved_state_dict['epoch']

        optimizer.load_state_dict(saved_state_dict['optimizer'])
        scheduler.load_state_dict(saved_state_dict['lr_schedule'])

    if params.batch_split_size is None or params.batch_split_size == 0:
        train_step_fn = training_step
    else:
        # Multi-staged training approach with large batch split into multiple smaller chunks with batch_split_size elems
        train_step_fn = multistaged_training_step



    ###########################################################################
    # Initialize Weights&Biases logging service
    ###########################################################################

    # params_dict = {e: params.__dict__[e] for e in params.__dict__ if e != 'model_params'}
    # model_params_dict = {"model_params." + e: params.model_params.__dict__[e] for e in params.model_params.__dict__}
    # params_dict.update(model_params_dict)
    # wandb.init(project='MinkLoc2', config=params_dict)

    ###########################################################################
    #
    ###########################################################################

    # Training statistics
    stats = {'train': [], 'eval': []}

    if 'val' in dataloaders:
        # Validation phase
        phases = ['train', 'val']
        # phases = ['val']
        stats['val'] = []
    else:
        phases = ['train']

    for epoch in tqdm.tqdm(range(start_epoch + 1, params.epochs + 1)):
        if epoch == start_epoch + 1:
            print('------cur_lr: {}------'.format(optimizer.param_groups[0]['lr']))
            print('------cur_epoch: {}------'.format(epoch))
        metrics = {'train': {}, 'val': {}}      # Metrics for wandb reporting

        for phase in phases:
            running_stats = []  # running stats for the current epoch and phase
            count_batches = 0

            if phase == 'train':
                global_iter = iter(dataloaders['train'])
            else:
                global_iter = None if dataloaders['val'] is None else iter(dataloaders['val'])

            while True:
                count_batches += 1
                batch_stats = {}
                if params.debug and count_batches > 2:
                    break

                try:
                    temp_stats = train_step_fn(global_iter, model, phase, device, optimizer, loss_fn)
                    batch_stats['global'] = temp_stats

                except StopIteration:
                    # Terminate the epoch when one of dataloders is exhausted
                    break

                running_stats.append(batch_stats)

            # Compute mean stats for the phase
            epoch_stats = {}
            for substep in running_stats[0]:
                epoch_stats[substep] = {}
                for key in running_stats[0][substep]:
                    temp = [e[substep][key] for e in running_stats]
                    if type(temp[0]) is dict:
                        epoch_stats[substep][key] = {key: np.mean([e[key] for e in temp]) for key in temp[0]}
                    elif type(temp[0]) is np.ndarray:
                        # Mean value per vector element
                        epoch_stats[substep][key] = np.mean(np.stack(temp), axis=0)
                    else:
                        epoch_stats[substep][key] = np.mean(temp)

            stats[phase].append(epoch_stats)
            print_stats(phase, epoch_stats)

            # Log metrics for wandb
            metrics[phase]['loss1'] = epoch_stats['global']['loss']
            if 'num_non_zero_triplets' in epoch_stats['global']:
                metrics[phase]['active_triplets1'] = epoch_stats['global']['num_non_zero_triplets']

            if 'positive_ranking' in epoch_stats['global']:
                metrics[phase]['positive_ranking'] = epoch_stats['global']['positive_ranking']

            if 'recall' in epoch_stats['global']:
                metrics[phase]['recall@1'] = epoch_stats['global']['recall'][1]

            if 'ap' in epoch_stats['global']:
                metrics[phase]['AP'] = epoch_stats['global']['ap']


        # ******* FINALIZE THE EPOCH *******

        # wandb.log(metrics)

        if scheduler is not None:
            scheduler.step()

        # if params.save_freq > 0 and epoch % params.save_freq == 0:
        #     torch.save(model.state_dict(), model_pathname + "_" + str(epoch) + ".pth")

        if epoch < 2 or (params.epochs < 100 and epoch >= 50) \
                or (params.epochs >= 300 and (epoch % 100 == 0 or epoch >= 380)):  # save mode and results
            model.eval()
            with torch.no_grad():
                prefix = "{}_epoch_{:0>3d}".format(model_name, epoch)
                final_eval_stats = evaluate(model, device, params, log=False)
                oxford_recall_1p, cur_mean_recall_1 = \
                    pnv_write_eval_stats('./' + model_name + '.txt', prefix, final_eval_stats)
                print('------oxford_recall_1p: {}, cur_mean_recall_1: {}------'
                      .format(oxford_recall_1p, cur_mean_recall_1))
                if epoch > 2:
                    final_model_path = "{}_epoch_{}".format(model_pathname, epoch) + '.pth'
                    checkpoint = {
                        "net": model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_schedule': scheduler.state_dict(),
                        'params': params,
                        "epoch": epoch
                    }
                    torch.save(checkpoint, final_model_path)
                    print('--------------save model: ' + final_model_path + '--------------')

        if params.batch_expansion_th is not None:
            # Dynamic batch size expansion based on number of non-zero triplets
            # Ratio of non-zero triplets
            le_train_stats = stats['train'][-1]  # Last epoch training stats
            rnz = le_train_stats['global']['num_non_zero_triplets'] / le_train_stats['global']['num_triplets']
            if rnz < params.batch_expansion_th:
                dataloaders['train'].batch_sampler.expand_batch()

    # print('')

    # Save final model weights
    # final_model_path = model_pathname + '_final.pth'
    # print(f"Saving weights: {final_model_path}")
    # torch.save(model.state_dict(), final_model_path)

    # Save final model weights
    # final_model_path = model_pathname + '_final.pth'

    # checkpoint = {
    #     "net": model.state_dict(),
    #     'optimizer': optimizer.state_dict(),
    #     'lr_schedule': scheduler.state_dict(),
    #     'params': params,
    #     "epoch": 0
    # }
    # torch.save(checkpoint, final_model_path)
    # torch.save(model.state_dict(), final_model_path)

    # Evaluate the final
    # PointNetVLAD datasets evaluation protocol
    # stats = evaluate(model, device, params, log=False)
    # print_eval_stats(stats)

    # print('.')

    # Append key experimental metrics to experiment summary file
    # model_params_name = os.path.split(params.model_params.model_params_path)[1]
    # config_name = os.path.split(params.params_path)[1]
    # model_name = os.path.splitext(os.path.split(final_model_path)[1])[0]
    # prefix = "{}, {}, {}".format(model_params_name, config_name, model_name)
    #
    # pnv_write_eval_stats("pnv_experiment_results.txt", prefix, stats)


def create_weights_folder():
    # Create a folder to save weights of trained models
    this_file_path = pathlib.Path(__file__).parent.absolute()
    temp, _ = os.path.split(this_file_path)
    weights_path = os.path.join(temp, 'weights')
    if not os.path.exists(weights_path):
        os.mkdir(weights_path)
    assert os.path.exists(weights_path), 'Cannot create weights folder: {}'.format(weights_path)
    return weights_path


def export_eval_stats(file_name, prefix, eval_stats):
    s = prefix
    with open(file_name, "a") as f:
        av_topN_recall = eval_stats['av_topN_recall']
        s += ", {:0.4f}, {:0.4f}, {:0.4f}, {:0.4f}, {:0.4f}\n"\
            .format(av_topN_recall[0], av_topN_recall[1], av_topN_recall[2], av_topN_recall[3], av_topN_recall[4])
        f.write(s)

