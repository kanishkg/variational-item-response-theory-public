import os
import time
import math
import numpy as np
from tqdm import tqdm
import sys

import torch
from torch import optim
import torch.distributions as dist
import torch.nn.functional as F

from src.torch_core.models import (
    VIBO_1PL, 
    VIBO_2PL, 
    VIBO_3PL,
    VIBO_STEP_1PL,
    VIBO_STEP_2PL,
    VIBO_STEP_3PL,
)
from src.datasets import load_dataset, artificially_mask_dataset, collate_function_step, artificially_mask_side_info, \
    create_encoder_mask
from src.utils import AverageMeter, save_checkpoint
from src.config import OUT_DIR, IS_REAL_WORLD
# from roar.pretraining import CharBERT, CharBERTClassifier
import environment

sys.path.append('../../socratic-tutor/')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--irt-model', type=str, default='1pl',
                        choices=['1pl', '2pl', '3pl'],
                        help='1pl|2pl|3pl (default: 1pl)')
    parser.add_argument('--dataset', type=str, default='1pl_simulation',
                        choices=[
                            '1pl_simulation', 
                            '2pl_simulation', 
                            '3pl_simulation',
                            '1pl_simulationstep',
                            '2pl_simulationstep',
                            '3pl_simulationstep',
                            'critlangacq',
                            'duolingo',
                            'wordbank',
                            'pisa2015_science',
                            'json',
                            'jsonstep',
                            'chessai',
                            'algebraai',
                            'roar',
                            'roarstep',
                            'abacus',
                            'abacusstep'
                        ],
                        help='which dataset to run on (default: 1pl_simulation)')
    parser.add_argument('--ability-dim', type=int, default=1,
                        help='number of ability dimensions (default: 1)')
    parser.add_argument('--ability-merge', type=str, default='product',
                        choices=['mean', 'product', 'transformer'],
                        help='mean|product|transformer (default: product)')
    parser.add_argument('--conditional-posterior', action='store_true', default=False,
                        help='q(ability|item,response) vs q(ability|response)')
    parser.add_argument('--generative-model', type=str, default='irt', 
                        choices=['irt', 'link', 'deep', 'residual'],
                        help='irt|link|deep|residual (default: irt)')
    parser.add_argument('--response-dist', type=str, default='bernoulli',
                        choices=['gaussian', 'bernoulli'],
                        help='gaussian|bernoulli (default: bernoulli)')
    parser.add_argument('--drop-missing', action='store_true', default=False)
    parser.add_argument('--artificial-missing-perc', type=float, default=0.,
                        help='how much to blank out so we can measure acc (default: 0)')
    parser.add_argument('--test-artificial-perc', type=float, default=0.,
                        help='how much to blank out so we can measure acc in test(default: 0)')
    parser.add_argument('--num-encode', type=int, default=-1,
                        help='number of responses to train with')
    parser.add_argument('--side-artificial-perc', type=float, default=0.,
                        help='how much to blank out so we can measure acc in test(default: 0)')

    parser.add_argument('--mask-items', action='store_true', default=False,
                        help='mask items in the train/test split (default: masks students)')
    parser.add_argument('--n-norm-flows', type=int, default=0,
                        help='Number of normalizing flows (default: 0)')

    parser.add_argument('--no-infer-dict', action='store_true', default=False,
                        help='if true, skip infer dict collection (default: False)')
    parser.add_argument('--no-marginal', action='store_true', default=False,
                        help='if true, skip marginal loglike computation (default: False)')
    parser.add_argument('--no-test', action='store_true', default=False,
                        help='if true, skip test (default: False)')
    parser.add_argument('--no-predictive', action='store_true', default=False,
                        help='if true, skip posterior predictive computation (default: False)')

    parser.add_argument('--no-test-predictive', action='store_true', default=False,
                        help='if true, skip posterior predictive computation (default: False)')

    parser.add_argument('--num-person', type=int, default=1000,
                        help='number of people in data (default: 1000)')
    parser.add_argument('--num-item', type=int, default=100,
                        help='number of people in data (default: 100)')
    parser.add_argument('--num-posterior-samples', type=int, default=400,
                        help='number of samples to use for analysis (default: 400)')
    parser.add_argument('--hidden-dim', type=int, default=64,
                        help='number of hidden dims (default: 64)')
    parser.add_argument('--max-num-person', help='limit the number of persons in dataset')
    parser.add_argument('--max-num-item', help='limit the number of items in dataset')

    parser.add_argument('--out-dir', type=str, default=OUT_DIR,
                        help='where to save chkpts (default: OUT_DIR)')

    parser.add_argument('--lr', type=float, default=5e-3,
                        help='default learning rate: 5e-3')
    parser.add_argument('--clip', type=float, default=10,
                        help='default gradient clip: 10')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 16)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--max-iters', type=int, default=-1, metavar='N',
                        help='number of maximum iterations (default: -1)')
    parser.add_argument('--num-workers', type=int, default=0, 
                        help='number of workers for data loading (default: 0)')
    parser.add_argument('--anneal-kl', action='store_true', default=False,
                        help='anneal KL divergence (default: False)')
    parser.add_argument('--beta-kl', type=float, default=1.0,
                        help='constant multiplier on KL (default: 1.0)')
    parser.add_argument('--predict', type=str,
                        help='File with words/nonwords to predict parameters.')

    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--gpu-device', type=int, default=0, 
                        help='which CUDA device to use (default: 0)')
    parser.add_argument('--embed-conpole', type=str, default=False,
                        help='Use the given pre-trained ConPoLe model to embed problems.')
    parser.add_argument('--side-info-model', type=str, default=False,
                        help='Use the given pre-trained  model to embed side-info.')
    parser.add_argument('--embed-bert', type=str, default=False,
                        help='Use the given pre-trained BERT model to embed problems.')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training (default: False)')
    args = parser.parse_args()

    side_info = None
    if args.n_norm_flows > 0:
        args.no_infer_dict = True
        args.no_predictive = True

    if args.artificial_missing_perc > 0:
        args.no_predictive = False

    if args.test_artificial_perc > 0:
        args.no_test_predictive = False

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if IS_REAL_WORLD[args.dataset]:
        # these params are only for IRT simulation datasets
        args.num_person = None
        args.num_item = None

        if args.max_num_person is not None:
            args.max_num_person = int(args.max_num_person)
        
        if args.max_num_item is not None:
            args.max_num_item = int(args.max_num_item)
        
    else:
        args.max_num_person = None
        args.max_num_item = None

    out_file = 'VIBO_{}_{}_{}_{}_{}person_{}item_{}maxperson_{}maxitem_{}maskperc_{}ability_{}_{}_{}seed{}_encode{}'.format(
        args.irt_model, 
        args.dataset,
        args.response_dist,
        args.generative_model,
        args.num_person, 
        args.num_item,
        args.max_num_person,
        args.max_num_item,
        args.artificial_missing_perc,
        args.ability_dim, 
        args.ability_merge,
        ('conpole_'
         if args.embed_conpole
         else ('bert'
               if args.embed_bert
               else '')),
        'conditional_q' if args.conditional_posterior else 'unconditional_q',
        args.seed,
        args.num_encode
    )
    args.out_dir = os.path.join(args.out_dir, out_file) 
    
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    device = torch.device("cuda" if args.cuda else "cpu")
    if args.cuda: torch.cuda.set_device(args.gpu_device)

    if args.response_dist == 'bernoulli':
        dataset_name = args.dataset
    else:
        dataset_name = f'{args.dataset}_continuous'

    if 'step' in args.dataset:
        side_info = True

    train_dataset = load_dataset(
        dataset_name, 
        train = True,
        num_person = args.num_person, 
        num_item = args.num_item,  
        ability_dim = args.ability_dim,
        max_num_person = args.max_num_person,
        max_num_item = args.max_num_item,
        side_info = side_info
    )
    test_dataset  = load_dataset(
        dataset_name, 
        train = False,
        num_person = args.num_person, 
        num_item = args.num_item, 
        ability_dim = args.ability_dim,
        max_num_person = args.max_num_person,
        max_num_item = args.max_num_item,
        side_info = side_info
    )

    if args.artificial_missing_perc > 0:
        train_dataset = artificially_mask_dataset(
            train_dataset,
            args.artificial_missing_perc,
            args.mask_items,
        )

    if args.test_artificial_perc > 0:
        test_dataset = artificially_mask_dataset(
            test_dataset,
            args.test_artificial_perc,
            args.mask_items,
        )

        if args.side_artificial_perc:
            train_dataset = artificially_mask_side_info(
                train_dataset,
                args.side_artificial_perc,
            )
    train_dataset = create_encoder_mask(
        train_dataset,
        args.num_encode
    )

    test_dataset = create_encoder_mask(
        test_dataset,
        args.num_encode
    )
    num_person = train_dataset.num_person
    num_item   = train_dataset.num_item

    collate_fn = None if 'step' not in args.dataset else collate_function_step

    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size = args.batch_size, 
        shuffle = True,
        num_workers = args.num_workers,
        collate_fn = collate_fn
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size = args.batch_size, 
        shuffle = False,
        num_workers = args.num_workers,
        collate_fn = collate_fn
    )
    N_mini_batches = len(train_loader)
    if args.max_iters != -1:
        args.epochs = int(math.ceil(args.max_iters / float(len(train_loader))))
        print(f'Found MAX_ITERS={args.max_iters}, setting EPOCHS={args.epochs}')

    if args.irt_model == '1pl':
        model_class = VIBO_1PL if 'step' not in args.dataset else VIBO_STEP_1PL
    elif args.irt_model == '2pl':
        model_class = VIBO_2PL if 'step' not in args.dataset else VIBO_STEP_2PL
    elif args.irt_model == '3pl':
        model_class = VIBO_3PL if 'step' not in args.dataset else VIBO_STEP_3PL
    else:
        raise Exception(f'model {args.irt_model} not recognized')

    if args.embed_conpole:
        embedding_model = torch.load(args.embed_conpole, map_location=device)
        embedding_model.to(device)
    elif args.embed_bert:
        embedding_model = torch.load(args.embed_bert, map_location=device)
        embedding_model.to(device)
    else:
        embedding_model = None
    side_info_model = args.side_info_model

    model = model_class(
        args.ability_dim,
        num_item,
        hidden_dim = args.hidden_dim,
        ability_merge = args.ability_merge,
        conditional_posterior = args.conditional_posterior,
        generative_model = args.generative_model,
        response_dist = args.response_dist,
        replace_missing_with_prior = not args.drop_missing,
        n_norm_flows = args.n_norm_flows,
        embedding_model = embedding_model,
        embed_conpole=args.embed_conpole,
        embed_bert=args.embed_bert,
        problems=train_dataset.problems,
        side_info_model=side_info_model,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    def get_annealing_factor(epoch, which_mini_batch):
        if args.anneal_kl:
            annealing_factor = \
                (float(which_mini_batch + epoch * N_mini_batches + 1) /
                 float(args.epochs // 2 * N_mini_batches))
        else:
            annealing_factor = args.beta_kl 
        return annealing_factor

    def train(epoch):
        model.train()
        train_loss = AverageMeter()
        grad_norm = AverageMeter()
        pbar = tqdm(total=len(train_loader))

        for batch_idx, batch in enumerate(train_loader):
            if 'step' in args.dataset:
                index, response, problem_ids, mask, steps, step_mask, encoder_mask = batch
                step_mask = step_mask.long().to(device)
            else:
                index, response, problem_ids, mask, encoder_mask = batch
            mb = response.size(0)
            response = response.to(device)
            mask = mask.long().to(device)
            encoder_mask = encoder_mask.long().to(device)
            annealing_factor = get_annealing_factor(epoch, batch_idx)
        
            optimizer.zero_grad()
            if args.n_norm_flows > 0:
                (
                    response, mask, response_mu, 
                    ability_k, ability, 
                    ability_mu, ability_logvar, ability_logabsdetjac, 
                    item_feat_k, item_feat, 
                    item_feat_mu, item_feat_logvar, item_feat_logabsdetjac,
                ) = model(response, mask)
                loss = model.elbo(
                    response, mask, response_mu, 
                    ability, ability_mu, ability_logvar,
                    item_feat, item_feat_mu, item_feat_logvar, 
                    annealing_factor = annealing_factor,
                    use_kl_divergence = False,
                    ability_k = ability_k,
                    item_feat_k = item_feat_k,
                    ability_logabsdetjac = ability_logabsdetjac,
                    item_logabsdetjac = item_feat_logabsdetjac,
                )
            else:
                if 'step' in args.dataset:
                     outputs = model(response, mask, steps, step_mask, encoder_mask)
                else:
                     outputs = model(response, mask, encoder_mask)
                loss = model.elbo(*outputs, annealing_factor=annealing_factor,
                                use_kl_divergence=True)/args.batch_size
            loss.backward()
            total_norm = -1
            weight_norm = 1

            if 'step' in args.dataset:
                for p in model.step_encoder.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.detach().data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()


            train_loss.update(loss.item(), mb)
            grad_norm.update(total_norm)

            pbar.update()
            pbar.set_postfix({'Loss': train_loss.avg, 'grad_step': grad_norm.avg})

        pbar.close()
        print('====> Train Epoch: {} Loss: {:.4f}'.format(epoch, train_loss.avg))

        return train_loss.avg

    def test(epoch):
        model.eval()
        test_loss = AverageMeter()
        pbar = tqdm(total=len(test_loader))

        with torch.no_grad():
            for batch in test_loader:
                if 'step' in args.dataset:
                    index, response, problem_ids, mask, steps, step_mask, encoder_mask = batch
                    step_mask = step_mask.long().to(device)
                else:
                    _, response, problem_ids, mask, encoder_mask = batch
                mb = response.size(0)
                response = response.to(device)
                mask = mask.long().to(device)
                encoder_mask = encoder_mask.long().to(device)

                if args.n_norm_flows > 0:
                    (
                        response, mask, response_mu, 
                        ability_k, ability, 
                        ability_mu, ability_logvar, ability_logabsdetjac, 
                        item_feat_k, item_feat, 
                        item_feat_mu, item_feat_logvar, item_feat_logabsdetjac,
                    ) = model(response, mask)
                    loss = model.elbo(
                        response, mask, response_mu, 
                        ability, ability_mu, ability_logvar,
                        item_feat, item_feat_mu, item_feat_logvar, 
                        use_kl_divergence = False,
                        ability_k = ability_k,
                        item_feat_k = item_feat_k,
                        ability_logabsdetjac = ability_logabsdetjac,
                        item_logabsdetjac = item_feat_logabsdetjac,
                    )
                else:
                    if 'step' in args.dataset:
                        outputs = model(response, mask, steps, step_mask, encoder_mask)
                    else:
                        outputs = model(response, mask, encoder_mask)
                    loss = model.elbo(*outputs)
                test_loss.update(loss.item(), mb)

                pbar.update()
                pbar.set_postfix({'Loss': test_loss.avg})

        pbar.close()
        print('====> Test Epoch: {} Loss: {:.4f}'.format(epoch, test_loss.avg))

        return test_loss.avg

    def get_log_marginal_density(loader):
        model.eval()
        meter = AverageMeter()
        pbar = tqdm(total=len(loader))

        with torch.no_grad():
            for _, response, _, mask in loader:
                mb = response.size(0)
                response = response.to(device)
                mask = mask.long().to(device)

                marginal = model.log_marginal(
                    response, 
                    mask, 
                    num_samples = args.num_posterior_samples,
                )
                marginal = torch.mean(marginal)
                meter.update(marginal.item(), mb)

                pbar.update()
                pbar.set_postfix({'Marginal': meter.avg})
        
        pbar.close()
        print('====> Marginal: {:.4f}'.format(meter.avg))

        return meter.avg

    def sample_posterior_predictive(loader):
        model.eval()
        meter = AverageMeter()
        pbar = tqdm(total=len(loader))

        with torch.no_grad():
            
            response_sample_set = []

            for batch in loader:
                if 'step' in args.dataset:
                    index, response, problem_ids, mask, steps, step_mask, encoder_mask = batch
                    step_mask = step_mask.long().to(device)
                else:
                    _, response, problem_ids, mask, encoder_mask = batch
                mb = response.size(0)
                response = response.to(device)
                mask = mask.long().to(device)
                encoder_mask = encoder_mask.long().to(device)

                if 'step' in args.dataset:
                    _, ability_mu, ability_logvar, _, item_feat_mu, item_feat_logvar, _, step_feat_mu, step_feat_logvar = \
                        model.encode(response, encoder_mask, steps, step_mask)
                else:
                    _, ability_mu, ability_logvar, _, item_feat_mu, item_feat_logvar = \
                        model.encode(response, encoder_mask)

                
                ability_scale = torch.exp(0.5 * ability_logvar)
                item_feat_scale = torch.exp(0.5 * item_feat_logvar)

                ability_posterior = dist.Normal(ability_mu, ability_scale)
                item_feat_posterior = dist.Normal(item_feat_mu, item_feat_scale)
                
                ability_samples = ability_posterior.sample([args.num_posterior_samples])
                item_feat_samples = item_feat_posterior.sample([args.num_posterior_samples])

                response_samples = []
                for i in range(args.num_posterior_samples):
                    ability_i = ability_samples[i]
                    item_feat_i = item_feat_samples[i]
                    response_i = model.decode(ability_i, item_feat_i).cpu()
                    response_samples.append(response_i)
                response_samples = torch.stack(response_samples)
                response_sample_set.append(response_samples)

                pbar.update()

            response_sample_set = torch.cat(response_sample_set, dim=1)

            pbar.close()

        return {'response': response_sample_set}

    def sample_posterior_mean(loader):
        model.eval()
        meter = AverageMeter()
        pbar = tqdm(total=len(loader))

        with torch.no_grad():
            
            response_sample_set = []

            for batch in loader:
                if 'step' in args.dataset:
                    index, response, problem_ids, mask, steps, step_mask, encoder_mask = batch
                    step_mask = step_mask.long().to(device)
                else:
                    _, response, problem_ids, mask, encoder_mask = batch
                mb = response.size(0)
                response = response.to(device)
                mask = mask.long().to(device)
                encoder_mask = encoder_mask.long().to(device)

                if 'step' in args.dataset:
                    _, ability_mu, ability_logvar, _, item_feat_mu, item_feat_logvar, _, step_feat_mu, step_feat_logvar = \
                        model.encode(response, encoder_mask, steps, step_mask)
                else:
                    _, ability_mu, ability_logvar, _, item_feat_mu, item_feat_logvar = \
                        model.encode(response, encoder_mask)

                
                response_sample = model.decode(ability_mu, item_feat_mu).cpu()
                response_sample_set.append(response_sample.unsqueeze(0))

                pbar.update()

            response_sample_set = torch.cat(response_sample_set, dim=1)

            pbar.close()

        return {'response': response_sample_set}

    def get_infer_dict(loader):
        model.eval()
        infer_dict = {}

        with torch.no_grad(): 
            ability_mus, item_feat_mus, step_feat_mus  = [], [], []
            ability_logvars, item_feat_logvars, step_feat_logvars = [], [], []

            pbar = tqdm(total=len(loader))
            for batch in loader:
                if 'step' in args.dataset:
                    _, response, _, mask, steps, step_mask, encoder_mask = batch
                    step_mask = step_mask.long().to(device)
                else:
                    _, response, _, mask, encoder_mask = batch
                mb = response.size(0)
                response = response.to(device)
                mask = mask.long().to(device)
                encoder_mask = encoder_mask.long().to(device)


                if 'step' in args.dataset:
                    _, ability_mu, ability_logvar, _, item_feat_mu, item_feat_logvar, _, step_feat_mu, step_feat_logvar = \
                        model.encode(response, encoder_mask, steps, step_mask)
                    step_feat_mus.append(step_feat_mu)
                    step_feat_logvars.append(step_feat_logvar)
                else:
                    _, ability_mu, ability_logvar, _, item_feat_mu, item_feat_logvar = \
                        model.encode(response, encoder_mask)

                ability_mus.append(ability_mu.cpu())
                ability_logvars.append(ability_logvar.cpu())

                item_feat_mus.append(item_feat_mu.cpu())
                item_feat_logvars.append(item_feat_logvar.cpu())

                if 'step' in args.dataset:
                    step_feat_mus.append(step_feat_mu.cpu())
                    step_feat_logvars.append(step_feat_logvar.cpu())

                pbar.update()

            ability_mus = torch.cat(ability_mus, dim=0)
            ability_logvars = torch.cat(ability_logvars, dim=0)
            pbar.close()

        infer_dict['ability_mu'] = ability_mus
        infer_dict['ability_logvar'] = ability_logvars
        infer_dict['item_feat_mu'] = item_feat_mu
        infer_dict['item_feat_logvar'] = item_feat_logvar
        if 'step' in args.dataset:
            infer_dict['step_feat_mu'] = step_feat_mu
            infer_dict['step_feat_logvar'] = step_feat_logvar


        return infer_dict

    is_best, best_loss = False, np.inf
    train_losses = np.zeros(args.epochs)
    if not args.no_test:
        test_losses  = np.zeros(args.epochs)
    train_times = np.zeros(args.epochs)

    for epoch in range(args.epochs):
        start_time = time.time()
        train_loss = train(epoch)
        end_time = time.time()
        train_losses[epoch] = train_loss
        train_times[epoch] = start_time - end_time
        
        if not args.no_test:
            test_loss = test(epoch)
            test_losses[epoch] = test_loss
            is_best = test_loss < best_loss
            best_loss = min(test_loss, best_loss)
        else:
            is_best = train_loss < best_loss
            best_loss = min(train_loss, best_loss)

        save_checkpoint({
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'args': args,
        }, is_best, folder=args.out_dir)

        np.save(os.path.join(args.out_dir, 'train_losses.npy'), train_losses)
        np.save(os.path.join(args.out_dir, 'train_times.npy'), train_times)

        if not args.no_test:
            np.save(os.path.join(args.out_dir, 'test_losses.npy'),  test_losses)

    for checkpoint_name in ['checkpoint.pth.tar', 'model_best.pth.tar']:
        checkpoint = torch.load(os.path.join(args.out_dir, checkpoint_name))
        model.load_state_dict(checkpoint['model_state_dict'])

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size = args.batch_size, 
            shuffle = False,
            collate_fn=collate_fn
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size = args.batch_size,
            shuffle = False,
            collate_fn=collate_fn
        )
        if not args.no_infer_dict:
            infer_dict = get_infer_dict(train_loader)
            checkpoint['infer_dict'] = infer_dict


        if not args.no_predictive:
            posterior_predict_samples = sample_posterior_predictive(train_loader)
            checkpoint['posterior_predict_samples'] = posterior_predict_samples

            if args.artificial_missing_perc > 0:
                missing_indices = train_dataset.missing_indices
                missing_labels = train_dataset.missing_labels

                if np.ndim(missing_labels) == 1:
                    missing_labels = missing_labels[:, np.newaxis]

                inferred_response = posterior_predict_samples['response'].mean(0)
                inferred_response = torch.round(inferred_response)

                correct, count = 0, 0
                print(inferred_response.size())
                for missing_index, missing_label in zip(missing_indices, missing_labels):
                    inferred_label = inferred_response[missing_index[0], missing_index[1]]
                    if inferred_label.item() == missing_label[0]:
                        correct += 1
                    count += 1
                missing_imputation_accuracy = correct / float(count)
                checkpoint['missing_imputation_accuracy'] = missing_imputation_accuracy
                model_name = "Amortized VIBO" if args.embed_bert or args.embed_conpole else "VIBO"
                if 'step' in args.dataset:
                    model_name = "Side-info A-VIBO" if args.embed_bert or args.embed_conpole else "Side-info VIBO"
                if not args.no_test_predictive:
                    posterior_predict_samples = sample_posterior_predictive(test_loader)
                    checkpoint['posterior_predict_samples'] = posterior_predict_samples

                    if args.test_artificial_perc > 0:
                        missing_indices = test_dataset.missing_indices
                        missing_labels = test_dataset.missing_labels

                        if np.ndim(missing_labels) == 1:
                            missing_labels = missing_labels[:, np.newaxis]

                        inferred_response = posterior_predict_samples['response'].mean(0)
                        inferred_response = torch.round(inferred_response)

                        correct, count = 0, 0
                        for missing_index, missing_label in zip(missing_indices, missing_labels):
                            inferred_label = inferred_response[missing_index[0], missing_index[1]]
                            if inferred_label.item() == missing_label[0]:
                                correct += 1
                            count += 1
                        test_missing_imputation_accuracy = correct / float(count)

                if 'best' in checkpoint_name:
                    with open('results_algebra','a') as f:
                        f.write(f'{{ "seed": {args.seed}, "model": "{model_name}","test_missing_perc": {args.test_artificial_perc}, "train_missing_perc": {args.artificial_missing_perc}, "train_accuracy": {missing_imputation_accuracy}, "test_accuracy": {test_missing_imputation_accuracy} , "num_encode": {args.num_encode}}},\n')
                print(f'{{ "seed": {args.seed}, "model": "{model_name}","test_missing_perc": {args.test_artificial_perc}, "train_missing_perc": {args.artificial_missing_perc}, "train_accuracy": {missing_imputation_accuracy}, "test_accuracy": {test_missing_imputation_accuracy} , "num_encode": {args.num_encode}}},')
                print(f'Missing Imputation Accuracy from samples: {missing_imputation_accuracy}')

            posterior_mean_samples = sample_posterior_mean(train_loader)
            
            if args.artificial_missing_perc > 0:
                missing_indices = train_dataset.missing_indices
                missing_labels = train_dataset.missing_labels

                if np.ndim(missing_labels) == 1:
                    missing_labels = missing_labels[:, np.newaxis]

                inferred_response = posterior_mean_samples['response'].squeeze(0)
                inferred_response = torch.round(inferred_response)

                correct, count = 0, 0
                for missing_index, missing_label in zip(missing_indices, missing_labels):
                    inferred_label = inferred_response[missing_index[0], missing_index[1]]
                    if inferred_label.item() == missing_label[0]:
                        correct += 1
                    count += 1
                missing_imputation_accuracy = correct / float(count)
                checkpoint['missing_imputation_accuracy_mean'] = missing_imputation_accuracy
                print(f'Missing Imputation Accuracy from mean: {missing_imputation_accuracy}')

        if not args.no_marginal:
            train_logp = get_log_marginal_density(train_loader)
            checkpoint['train_logp'] = train_logp

            if not args.no_test:
                test_logp = get_log_marginal_density(test_loader)
                checkpoint['test_logp'] = test_logp

        if not args.no_test_predictive and args.no_predictive:
            posterior_predict_samples = sample_posterior_predictive(test_loader)
            checkpoint['posterior_predict_samples'] = posterior_predict_samples

            if args.test_artificial_perc > 0:
                missing_indices = test_dataset.missing_indices
                missing_labels = test_dataset.missing_labels

                if np.ndim(missing_labels) == 1:
                    missing_labels = missing_labels[:, np.newaxis]

                inferred_response = posterior_predict_samples['response'].mean(0)
                inferred_response = torch.round(inferred_response)

                correct, count = 0, 0
                for missing_index, missing_label in zip(missing_indices, missing_labels):
                    inferred_label = inferred_response[missing_index[0], missing_index[1]]
                    if inferred_label.item() == missing_label[0]:
                        correct += 1
                    count += 1
                missing_imputation_accuracy = correct / float(count)
                checkpoint['missing_imputation_accuracy'] = missing_imputation_accuracy
                model_name = "Amortized VIBO" if args.embed_bert or args.embed_conpole else "VIBO"
                if 'step' in args.dataset:
                    model_name = "Side-info A-VIBO" if args.embed_bert or args.embed_conpole else "Side-info VIBO"
                print(f'{{ "seed": {args.seed}, "model": "{model_name}","missing side": {args.side_artificial_perc}, "missing_perc": {args.test_artificial_perc}, "test_accuracy": {missing_imputation_accuracy} }},')
                print(f'Missing Imputation Accuracy from samples: {missing_imputation_accuracy}')

        torch.save(checkpoint, os.path.join(args.out_dir, checkpoint_name))
        print(f'Train time: {np.abs(train_times[:100]).sum()}')
