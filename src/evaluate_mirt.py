import json

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
from scipy import stats

import torch
import torch.distributions.constraints as constraints

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO

from py_irt.models.four_param_logistic import FourParamLog
from py_irt.dataset import Dataset


from rich.console import Console
from rich.live import Live
from rich.table import Table


from src.datasets import *
from src.utils import *
from src.evaluate_transfer import *

def predict(ability, discs, diffs):
    a =  -np.array(discs) * (np.array(ability) - np.array(diffs))
    a = a.sum(axis=-1)
    return (1 / (1 + np.exp(a)))

def model_hierarchical(subjects, items, obs, params, num_person=39, dims=3):
    with pyro.plate("mu_theta_plate", dims):
        mu_theta = pyro.sample(
        "mu_theta",
        dist.Normal(
            torch.tensor(0.0),
            torch.tensor(1.0e6),
        ),
    )
    with pyro.plate("u_theta_plate", dims):

        u_theta = pyro.sample(
            "u_theta",
            dist.Gamma(
                torch.tensor(1.0),
                torch.tensor(1.0),
            ),
        )
    with pyro.plate("thetas", num_person, dim=-2):
        with pyro.plate("theta_dims", dims, dim=-1):
            ability = pyro.sample("theta", dist.Normal(mu_theta, 1.0 / u_theta))
    diff = torch.tensor(params['diff'])
    disc = torch.tensor(params['disc'])
    with pyro.plate("observe_data", obs.size(0)):
        multidim_logits = disc[items] * (ability[subjects] - diff[items])
        logits = multidim_logits.sum(axis=-1)
        pyro.sample("obs", dist.Bernoulli(logits=logits), obs=obs)
    
def guide_hierarchical(subjects, items, obs, params, num_person=39, dims=3):
    loc_mu_theta_param = pyro.param("loc_mu_theta", torch.zeros(dims))
    scale_mu_theta_param = pyro.param(
        "scale_mu_theta",
        1e2 * torch.ones(dims),
        constraint=constraints.positive,
    )
    alpha_theta_param = pyro.param(
        "alpha_theta",
        torch.ones(dims),
        constraint=constraints.positive,
    )
    beta_theta_param = pyro.param(
        "beta_theta",
        torch.ones(dims),
        constraint=constraints.positive,
    )

    # sample statements
    m_theta_param = pyro.param(
        "loc_ability", torch.zeros([num_person, dims])
    )
    s_theta_param = pyro.param(
        "scale_ability",
        torch.ones([num_person, dims]),
        constraint=constraints.positive,
    )

    with pyro.plate("mu_theta_plate", dims):
        mu_theta = pyro.sample(
            "mu_theta", dist.Normal(loc_mu_theta_param, scale_mu_theta_param)
        )
    with pyro.plate("u_theta_plate", dims):
        u_theta = pyro.sample("u_theta", dist.Gamma(alpha_theta_param, beta_theta_param))
    with pyro.plate("thetas", num_person, dim=-2):
        with pyro.plate("theta_dims", dims, dim=-1):
            theta = pyro.sample("theta", dist.Normal(m_theta_param, s_theta_param))

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample-choice', type=str, default='random')
    parser.add_argument('--num-seed', type=int, default=1)

    args = parser.parse_args()

    seed_array = list(range(args.num_seed-1)) + [42]
    seed_array = [16, 17, 18, 19, 42]
    # number of samples for the encoder
    dataset_encode = list(range(1,11)) + [-1]

    # best item params
    best_params = './data/py_irt/best_parameters.json'
    params = {}
    with open(best_params) as f:
        data = f.read()
        params = json.loads(data)
        
    # load datasets
    test_dataset = load_dataset('json', train=False)
    train_dataset = load_dataset('json', train=True)

    # join datasets
    train_dataset.response = np.concatenate([train_dataset.response, test_dataset.response], axis=0)
    train_dataset.mask = np.concatenate([train_dataset.mask, test_dataset.mask], axis=0)
    train_dataset.num_person = train_dataset.num_person + test_dataset.num_person


    # choose sampling strategy for the encoder
    if args.sample_choice == "disc":
        item_param = np.array(params['disc'])
        encoder_mask_fn = disc_encoder_mask
    elif args.sample_choice == "random":
        item_param = None
        encoder_mask_fn = create_encoder_mask
    elif args.sample_choice == "difficulty":
        item_param = np.array(params['diff'])
        encoder_mask_fn = diff_encoder_mask

    # calculate empirical ability
    total_score = train_dataset.response[:,:,0].sum(1)
    total_attempts = train_dataset.mask[:,:,0].sum(1)
    empirical_ability = total_score/total_attempts    
    for seed in seed_array:
        for num_encode in tqdm(dataset_encode):
            pyro.clear_param_store()
            torch.manual_seed(seed)
            np.random.seed(seed)
            pyro.set_rng_seed(seed)

            train_dataset_masked = artificially_mask_dataset(train_dataset, 0.1, seed) 
            # create encoder mask
            if args.sample_choice == "random":
                train_dataset_masked =  encoder_mask_fn(train_dataset_masked, num_encode, seed)
            else:
                train_dataset_masked = encoder_mask_fn(train_dataset_masked, num_encode, item_param.sum(-1))
            
            # build inputs for mirt
            subjects = []
            items = []
            obs = []
            for i in range(train_dataset_masked.response.shape[0]):
                for j in range(train_dataset_masked.response.shape[1]):
                    if train_dataset_masked.encoder_mask[i,j,0] == 1:
                        subjects.append(i)
                        items.append(j)
                        obs.append(train_dataset_masked.response[i,j,0])
            subjects = torch.tensor(subjects, dtype=torch.long)
            items = torch.tensor(items, dtype=torch.long)
            obs = torch.tensor(obs, dtype=torch.float)
            
            # fit mirt on ability
            scheduler = pyro.optim.ExponentialLR(
            {
                "optimizer": torch.optim.Adam,
                "optim_args": {"lr": 1e-1},
                "gamma":  0.9999,
            }
                )
            svi = SVI(model_hierarchical, guide_hierarchical, scheduler, loss=Trace_ELBO())
            _ = model_hierarchical(subjects, items, obs, params, train_dataset.num_person)
            _ = guide_hierarchical(subjects, items, obs, params, train_dataset.num_person)

            
            table = Table()
            epochs = 2000
            table.add_column("Epoch")
            table.add_column("Loss")
            table.add_column("Best Loss")
            table.add_column("New LR")
            loss = float("inf")
            best_loss = loss
            responses = obs
            current_lr = 1e-1
            with Live(table) as live:
                live.console.print(f"Training Pyro IRT Model for {epochs} epochs")
                for epoch in range(epochs):
                    loss = svi.step(subjects, items, responses, params, train_dataset_masked.num_person)
                    if loss < best_loss:
                        best_loss = loss
                        best_ability = pyro.param("loc_ability").data
                    scheduler.step()
                    current_lr = current_lr * 0.9999
                    if epoch % 100 == 0:
                        table.add_row(
                            f"{epoch + 1}", "%.4f" % loss, "%.4f" % best_loss, "%.4f" % current_lr
                        )

                table.add_row(f"{epoch + 1}", "%.4f" % loss, "%.4f" % best_loss, "%.4f" % current_lr)

            # get best params
            ability_predicted = best_ability.sum(-1)
            # get metrics
            r = stats.stats.pearsonr(ability_predicted, empirical_ability)[0]
            missing_indices = train_dataset_masked.missing_indices
            missing_labels = train_dataset_masked.missing_labels
            predicted = []
            actual = []
            for missing_index, missing_label in zip(missing_indices, missing_labels):
                p, q = missing_index
                inferred_label = predict(ability_predicted[p], params['disc'][q], params['diff'][q])
                actual.append(missing_label[0])
                predicted.append(inferred_label.item())
            metrics = evaluate_metrics(actual, predicted)
            acc = metrics['accuracy']
            auroc = metrics['auroc']
            f1 = metrics['F1']
            # write to file
            out_file = f'mirt_{args.sample_choice}.csv'
            with open(out_file, 'a') as f:
                f.write(f'{seed},{num_encode},{r},{acc},{auroc},{f1}\n')
            del(train_dataset_masked)
    