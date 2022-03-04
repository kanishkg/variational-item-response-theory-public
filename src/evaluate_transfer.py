import argparse
from ast import Num
import sys

import torch
import seaborn as sns
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.datasets import *
from src.torch_core.models import (
    VIBO_1PL, 
    VIBO_2PL, 
    VIBO_3PL,
    VIBO_STEP_1PL, 
    VIBO_STEP_2PL, 
    VIBO_STEP_3PL,   
    
)
from utils import evaluate_metrics

sys.path.append('../../socratic-tutor/')

def get_infer_dict(loader, model, step=0):
    model.eval()
    infer_dict = {}
    with torch.no_grad(): 
        ability_mus, item_feat_mus, step_feat_mus  = [], [], []
        ability_logvars, item_feat_logvars, step_feat_logvars = [], [], []

        for batch in loader:
            if step:
                _, response, _, mask, steps, step_mask, encoder_mask = batch
                step_mask = step_mask.long().to(device)
            else:
                _, response, _, mask, encoder_mask = batch
            mb = response.size(0)
            response = response.to(device)
            mask = mask.long().to(device)
            encoder_mask = encoder_mask.long().to(device)

            if step:
                _, ability_mu, ability_logvar, _, item_feat_mu, item_feat_logvar, _, step_feat_mu, step_feat_logvar = \
                    model.encode(response, encoder_mask, steps, step_mask)
            else:
                _, ability_mu, ability_logvar, _, item_feat_mu, item_feat_logvar = \
                model.encode(response, encoder_mask)

            ability_mus.append(ability_mu.cpu())
            ability_logvars.append(ability_logvar.cpu())

            item_feat_mus.append(item_feat_mu.cpu())
            item_feat_logvars.append(item_feat_logvar.cpu())
        ability_mus = torch.cat(ability_mus, dim=0)
        ability_logvars = torch.cat(ability_logvars, dim=0)
    return ability_mus, item_feat_mus

def disc_encoder_mask(old_dataset, num_encode, item_disc):
    if num_encode == -1:
        old_dataset.encoder_mask = old_dataset.mask
        return old_dataset
    dataset = copy.deepcopy(old_dataset)
    mask = dataset.mask
    encoder_mask = np.zeros((mask.shape[0], mask.shape[1], 1))
    # iterate over students and randomly choose num encode samples
    for i in range(mask.shape[0]):
        cols = np.where(mask[i, :, 0] != 0)[0]        
        # pass when attempted is less than num_encode
        if cols.shape[0] < num_encode:
            continue
        item_disc_n = item_disc[cols]
        items = item_disc_n.argsort()[-num_encode:][::-1]
        encoder_mask[i, cols[items], 0] = 1
    dataset.encoder_mask = encoder_mask
    return dataset

def diff_encoder_mask(old_dataset, num_encode, item_diff):
    if num_encode == -1:
        old_dataset.encoder_mask = old_dataset.mask
        return old_dataset
    dataset = copy.deepcopy(old_dataset)
    mask = dataset.mask
    encoder_mask = np.zeros((mask.shape[0], mask.shape[1], 1))
    # iterate over students and randomly choose num encode samples
    for i in range(mask.shape[0]):
        cols = np.where(mask[i, :, 0] != 0)[0]
        # pass when attempted is less than num_encode
        if cols.shape[0] < num_encode:
            continue
        item_diff_n = item_diff[cols]
        item_ids = item_diff_n.argsort()
        for j in range(num_encode):
            item_idx = item_ids[int(len(item_ids)/2)]
            encoder_mask[i, cols[item_idx], 0] = 1
            response = dataset.response[i, cols[item_idx], 0]
            if response == 1:
                item_ids = item_ids[int(len(item_ids)/2)+1:]
            elif response == 0:
                item_ids = item_ids[:int(len(item_ids)/2)]
            if len(item_ids) == 0:
                break
    dataset.encoder_mask = encoder_mask
    return dataset

def create_encoder_mask(old_dataset, num_encode, seed):
    if num_encode == -1:
        old_dataset.encoder_mask = old_dataset.mask
        return old_dataset
    dataset = copy.deepcopy(old_dataset)
    mask = dataset.mask
    encoder_mask = np.zeros((mask.shape[0], mask.shape[1], 1))
    rs = np.random.RandomState(seed)
    # iterate over students and randomly choose num encode samples
    for i in range(mask.shape[0]):
        cols = np.where(mask[i, :, 0] != 0)[0]
        if cols.shape[0] < num_encode:
            continue
        items = rs.choice(cols, size=num_encode, replace=False)
        encoder_mask[i, items, 0] = 1
    dataset.encoder_mask = encoder_mask
    return dataset

def sample_posterior_mean(model, loader, step=0):
    model.eval()
    device = torch.device("cuda")
    with torch.no_grad():
        response_sample_set = []
        for batch in loader:
            if step:
                _, response, _, mask, steps, step_mask, encoder_mask = batch
                step_mask = step_mask.long().to(device)
            else:
                _, response, _, mask, encoder_mask = batch
            mb = response.size(0)
            response = response.to(device)
            mask = mask.long().to(device)
            encoder_mask = encoder_mask.long().to(device)
            if step:
                _, ability_mu, ability_logvar, _, item_feat_mu, item_feat_logvar, _, step_feat_mu, step_feat_logvar = \
                    model.encode(response, encoder_mask, steps, step_mask)
            else:
                _, ability_mu, ability_logvar, _, item_feat_mu, item_feat_logvar = \
                model.encode(response, encoder_mask)

            response_sample = model.decode(ability_mu, item_feat_mu).cpu()
            response_sample_set.append(response_sample.unsqueeze(0))

        response_sample_set = torch.cat(response_sample_set, dim=1)
    return response_sample_set

def get_missing(dataset):
    response = dataset.response
    mask = dataset.mask
    encoder_mask = dataset.encoder_mask
    missing_indices = []
    missing_labels = []
    for i in range(mask.shape[0]):
        cols = np.where(mask[i, :, 0] != 0)[0]
        cols_encoder = np.where(encoder_mask[i, :, 0] != 0)[0]
        missing_indices += [[i, c] for c in cols if c not in cols_encoder]
        missing_labels += [response[i, c, 0] for c in cols if c not in cols_encoder]
    return missing_indices, missing_labels

def artificially_mask_dataset(old_dataset, perc, seed, mask_items=False):
    dataset = copy.deepcopy(old_dataset)
    assert perc >= 0 and perc <= 1
    response = dataset.response
    mask = dataset.mask

    if np.ndim(mask) == 2:
        row, col = np.where(mask != 0)
    elif np.ndim(mask) == 3:
        row, col = np.where(mask[:, :, 0] != 0)
    pool = np.array(list(zip(row, col)))
    num_all = pool.shape[0]
    rs = np.random.RandomState(seed)
    labels = []

    if not mask_items:
        # As before, just choose a random subset of the labels.
        num = int(perc * num_all)
        indices = np.sort(
            rs.choice(np.arange(num_all), size=num, replace=False),
        )
        label_indices = pool[indices]

        for idx in label_indices:
            label = copy.deepcopy(response[idx[0], idx[1]])
            labels.append(label)
            mask[idx[0], idx[1]] = 0
            response[idx[0], idx[1]] = -1
    else:
        # First choose a random subset of the items, then mask all of their labels.
        num = int(perc * len(dataset.problems))
        items = np.sort(
            rs.choice(np.arange(len(dataset.problems)),
                      size=num, replace=False),
        )
        for item in items:
            mask[dataset.problem_id == item] = 0

        (rows, cols, _) = np.nonzero(1 - mask)
        label_indices = np.stack([rows, cols], axis=1)

        for r, c in zip(rows, cols):
            label = copy.deepcopy(response[r, c])
            labels.append(label)
            response[r, c] = -1

    labels = np.array(labels)

    dataset.response = response
    dataset.mask = mask
    dataset.missing_labels = labels
    dataset.missing_indices = label_indices

    return dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-enc', type=int, default=20)
    parser.add_argument('--sample-choice', type=str, default='random')
    parser.add_argument('--model-name', type=str, default='empirical')
    parser.add_argument('--num-seed', type=int, default=1)
    parser.add_argument('--step', action='store_true', default=False)
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training (default: False)')
    parser.add_argument('--gpu', type=int, default=0)

    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda else "cpu") 
    if args.cuda:
        torch.cuda.set_device(args.gpu)
    torch.manual_seed(1)
    np.random.seed(1)

    seed_array = list(range(args.num_seed))

    # number of samples for the encoder
    dataset_encode = list(range(1,11)) + [-1]
    
    # load checkpoints 
    if not args.step:
        vibo_all = torch.load(f'./out/VIBO_2pl_algebraai_bernoulli_irt_Noneperson_Noneitem_Nonemaxperson_Nonemaxitem_0.1maskperc_1ability_product__conditional_qseed1_encode{args.model_enc}/model_best.pth.tar',map_location=device)
    elif args.step:
        vibo_all = torch.load(f'./out/VIBO_2pl_algebraaistep_bernoulli_irt_Noneperson_Noneitem_Nonemaxperson_Nonemaxitem_0.1maskperc_1ability_product__conditional_qseed1_encode{args.model_enc}/model_best.pth.tar',map_location=device)

    
    # load datasets
    if not args.step:
        test_dataset = load_dataset('json', train=False)
        train_dataset = load_dataset('json', train=True)
    elif args.step:
        test_dataset = load_dataset('jsonstep', train=False)
        train_dataset = load_dataset('jsonstep', train=True)
        
    # load model weights and initialize
    side_info_model = 'conpole_trajectory'
    if not args.step:
        model = VIBO_2PL(
            1,
            788,
            hidden_dim = 64,
            ability_merge = 'product',
            conditional_posterior = True,
            generative_model = 'irt',
            response_dist = 'bernoulli',
            replace_missing_with_prior = False,
            n_norm_flows = False,
            embedding_model = None,
            embed_conpole=None,
            embed_bert=None,
            problems=test_dataset.problems,
            side_info_model=None,
            device=device
        ).to(device)
    elif args.step:
        model = VIBO_STEP_2PL(
            1,
            788,
            hidden_dim = 64,
            ability_merge = 'product',
            conditional_posterior = True,
            generative_model = 'irt',
            response_dist = 'bernoulli',
            replace_missing_with_prior = False,
            n_norm_flows = False,
            embedding_model = None,
            embed_conpole=None,
            embed_bert=None,
            problems=test_dataset.problems,
            side_info_model=side_info_model,
            device=device
        ).to(device)
    
    model.load_state_dict(vibo_all['model_state_dict'])
    model.eval()

    # choose sampling strategy for the encoder
    if args.sample_choice == "disc":
        item_domain = torch.arange(788).unsqueeze(1).to(device)
        mu, _ = model.item_encoder(item_domain)
        item_param = np.array([mu[i, 0].item() for i in range(788)])
        encoder_mask_fn = disc_encoder_mask
    elif args.sample_choice == "random":
        item_param = None
        encoder_mask_fn = create_encoder_mask
    elif args.sample_choice == "difficulty":
        item_domain = torch.arange(788).unsqueeze(1).to(device)
        mu, _ = model.item_encoder(item_domain)
        item_param = np.array([mu[i, 1].item() for i in range(788)])
        encoder_mask_fn = diff_encoder_mask

    # calculate empirical ability
    total_score = test_dataset.response[:,:,0].sum(1)
    total_attempts = test_dataset.mask[:,:,0].sum(1)
    empirical_ability_test = total_score/total_attempts
    total_score = train_dataset.response[:,:,0].sum(1)
    total_attempts = train_dataset.mask[:,:,0].sum(1)
    empirical_ability_train = total_score/total_attempts
    empirical_ability = np.concatenate((empirical_ability_train, empirical_ability_test))

    for seed in seed_array:
        train_dataset_masked = artificially_mask_dataset(train_dataset, 0.1, seed) 
        test_dataset_masked = artificially_mask_dataset(test_dataset, 0.1, seed) 
        for num_encode in tqdm(dataset_encode):
            # create encoder mask
            if args.sample_choice == "random":
                train_dataset_masked =  encoder_mask_fn(train_dataset_masked, num_encode, seed)
                test_dataset_masked = encoder_mask_fn(test_dataset_masked, num_encode, seed)
            else:
                train_dataset_masked = encoder_mask_fn(train_dataset_masked, num_encode, item_param)
                test_dataset_masked = encoder_mask_fn(test_dataset_masked, num_encode, item_param)

            # get missing data when using unseen samples from encoder
#             missing_indices_train, missing_labels_train = get_missing(train_dataset_masked)
#             missing_indices_test, missing_labels_test = get_missing(test_dataset_masked)

            # get missing data when artificially masking
            missing_indices_train, missing_labels_train = train_dataset_masked.missing_indices, train_dataset_masked.missing_labels
            missing_indices_test, missing_labels_test = test_dataset_masked.missing_indices, test_dataset_masked.missing_labels

            missing_labels = missing_labels_train.squeeze().tolist() + missing_labels_test.squeeze().tolist()

            if args.model_name == 'empirical': 
                seen_response_test = test_dataset_masked.response * test_dataset_masked.encoder_mask
                empirical_estimate_test = (seen_response_test.sum(1)/test_dataset_masked.encoder_mask.sum(1)).squeeze()
                seen_response_train = train_dataset_masked.response * train_dataset_masked.encoder_mask
                empirical_estimate_train = (seen_response_train.sum(1)/train_dataset_masked.encoder_mask.sum(1)).squeeze()
                ability_predicted = np.concatenate((empirical_estimate_train, empirical_estimate_test))

                inferred_response_train = np.round(np.tile(empirical_ability_train, (test_dataset_masked.response.shape[1], 1))).T
                inferred_response_test = np.round(np.tile(empirical_ability_test, (test_dataset_masked.response.shape[1], 1))).T
                inferred_labels_train = [inferred_response_train[x, y] for x, y in missing_indices_train]
                inferred_labels_test = [inferred_response_test[x, y] for x, y in missing_indices_test]
                inferred_labels = inferred_labels_train + inferred_labels_test


            elif args.model_name == 'vibo':
                if not args.step:
                    test_loader = torch.utils.data.DataLoader(
                            test_dataset_masked,
                            batch_size = 16,
                            shuffle = False,
                            collate_fn=None
                        )
                    train_loader = torch.utils.data.DataLoader(
                            train_dataset_masked,
                            batch_size = 16,
                            shuffle = False,
                            collate_fn=None
                        )
                elif args.step:
                    test_loader = torch.utils.data.DataLoader(
                                test_dataset_masked,
                                batch_size = 16,
                                shuffle = False,
                                collate_fn=collate_function_step
                            )
                    train_loader = torch.utils.data.DataLoader(
                                train_dataset_masked,
                                batch_size = 16,
                                shuffle = False,
                                collate_fn=collate_function_step
                            )
                vibo_ability_train, vibo_item = get_infer_dict(train_loader, model, args.step)
                vibo_ability_test, vibo_item = get_infer_dict(test_loader, model, args.step)
                ability_predicted = torch.cat([vibo_ability_train[:, 0],vibo_ability_test[:, 0]]) .cpu().numpy()
                response_set_train = sample_posterior_mean(model, train_loader, args.step).squeeze()
                response_set_test = sample_posterior_mean(model, test_loader, args.step).squeeze()
                inferred_response_train = torch.round(response_set_train).cpu().numpy()
                inferred_response_test = torch.round(response_set_test).cpu().numpy()
                inferred_labels_train = [inferred_response_train[x, y] for x, y in missing_indices_train]
                inferred_labels_test = [inferred_response_test[x, y] for x, y in missing_indices_test]
                inferred_labels = inferred_labels_train + inferred_labels_test

                
            # calculate ability score
            r = stats.stats.pearsonr(ability_predicted, empirical_ability)[0]

            # calculate accuracy, auROC, and F1
            if num_encode == -1 and args.model_name == 'empirical':
                acc = 1.
                auroc = 1.
                f1 = 1.
            else:
                missing_indices = train_dataset_masked.missing_indices
                missing_labels = train_dataset_masked.missing_labels
                predicted = []
                actual = []
                for missing_index, missing_label in zip(missing_indices, missing_labels):
                    inferred_label = inferred_response_train[missing_index[0],
                                            missing_index[1]]
                    actual.append(missing_label[0])
                    predicted.append(inferred_label.item())
                missing_indices = test_dataset_masked.missing_indices
                missing_labels = test_dataset_masked.missing_labels


                for missing_index, missing_label in zip(missing_indices, missing_labels):
                    inferred_label = inferred_response_test[missing_index[0],
                                                    missing_index[1]]
                    actual.append(missing_label[0])
                    predicted.append(inferred_label.item())

                metrics = evaluate_metrics(missing_labels, inferred_labels)
                acc = metrics['accuracy']
                auroc = metrics['auroc']
                f1 = metrics['F1']
           
            # write to file
            out_file = f'{args.model_name}_{args.model_enc}_{args.step}_{args.sample_choice}.csv'
            with open(out_file, 'a') as f:
                f.write(f'{seed},{num_encode},{r},{acc},{auroc},{f1}\n')