import os
import time
import math
import numpy as np
from tqdm import tqdm
import sys

import torch

from src.datasets import load_dataset, artificially_mask_dataset, collate_function_step
import environment
import argparse

sys.path.append('../../socratic-tutor/')


def get_algebra_data(num_states=None):
    train_dataset = load_dataset('json', is_train=True)
    if num_states is None:
        num_states = train_dataset.n_problems
    problem_states = [environment.State([p], [], 0) for p in train_dataset.problems[:num_states]]
    return problem_states


def evaluate_solver(problems, checkpoint, beam_size, max_steps, debug=False):
    model = torch.load(checkpoint, map_location=device)
    model.to(device)
    env = environment.RustEnvironment("equations-ct")
    responses = []
    scores = 0.
    total = 0.
    pbar = tqdm(problems)

    for state in pbar:
        total += 1
        success, history = model.rollout(env, state,
                                         max_steps, beam_size, )
        if success:
            scores += 1
            responses.append(1)
        else:
            responses.append(0)
        pbar.set_description(f"current score {scores / total}")
    return responses, history


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--population-type', type=str, default='epoch',
                        choices=['epoch', 'depth', 'beam'],
                        help='epoch|depth|beam (default: epoch)')
    parser.add_argument('--ckpt-path', type=str, default='/mnt/fs3/poesia/socratic-tutor/output/algebra-solver/ConPoLe/equations-ct/run0/checkpoints/',
                        help='path to checkpoints')
    parser.add_argument('--save-path', type=str, default='/mnt/fs1/kanishkg/rich-irt/variational-item-response-theory-public/data/algebra',
                        help='path where to save the results')
    parser.add_argument('--save-file', type=str, default='algebra_steps.pth',
                        help='name of the file to save the results')
    parser.add_argument('--max-depth', type=int, default=30,
                        help='maximum depth of search')
    parser.add_argument('--beam-size', type=int, default=5,
                        help='size of beam search')
    parser.add_argument('--best-epoch', type=int, default=88,
                        help='number of the best epoch')
    parser.add_argument('--num-states', type=int, default=None,
                        help='number of problems to evaluate')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='whether to print debug info like steps')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='whether to use cuda')    
    args = parser.parse_args()

    if args.population_type == 'beam-size':
        population_parameters = {'beam-size': reversed([i + 1 for i in range(18)])}
    elif args.population_type == 'epoch':
        best_epoch = 88
        population_parameters = {'epoch': [args.best_epoch - i for i in range(30)]}
    elif args.population_type == 'depth':
        population_parameters = {'depth': [i + 1 for i in range(args.max_depth)]}

    device = torch.device("cuda" if args.cuda else "cpu")

    train_dataset = load_dataset('json', is_train=True)
    problem_states = get_algebra_data(args.num_states)
    responses = []
    dataset = {'response': [], 'epoch': [], 'beam': [], 'depth': [], 'score':[], 'steps':[],
                   'problems': train_dataset.problems}

    for p in population_parameters[args.population_type]:
        epoch = args.best_epoch
        depth = args.max_depth
        beam = args.beam_size
        if args.population_type == 'beam-size':
            res, steps = evaluate_solver(problem_states, os.path.join(args.ckpt_path, f'{args.ckpt}.pt'), p, args.max_depth, args.debug)
            beam = p
        elif args.population_type == 'epoch':
            res, steps = evaluate_solver(problem_states, os.path.join(args.ckpt_path, f'{p}.pt'), args.beam_size, args.max_depth, args.debug)
            epoch = p
        elif args.population_type == 'depth':
            res, steps = evaluate_solver(problem_states, os.path.join(args.ckpt_path, f'{args.ckpt}.pt'), args.beam_size, p, args.debug)
            depth = p
        dataset['response'].append(res)
        dataset['epoch'].append(epoch)
        dataset['beam'].append(beam)
        dataset['depth'].append(depth)
        dataset['score'].append(sum(res)/len(res))
        dataset['steps'].append(steps)
        print(f"epoch: {epoch}, beam: {beam}, depth: {depth}, score: {sum(res)/len(res)}")
        torch.save(dataset, os.path.join(args.save_path,
                                     args.save_file))
