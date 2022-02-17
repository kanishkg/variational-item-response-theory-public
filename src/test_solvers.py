import os
import time
import random
import math
import numpy as np
from tqdm import tqdm
import sys
import subprocess

import torch

from src.datasets import load_dataset, artificially_mask_dataset, collate_function_step
import environment
import argparse

sys.path.append('../../socratic-tutor/')

signs = ['+', '-']
symbols = ['(', ')', ' ']

def corrupt_state(state):
    final_fact = state.facts[-1]
    sigs = [(i, s) for i, s in enumerate(final_fact) if s in signs]
    if len(sigs) == 0:
        return state
    random.shuffle(sigs)
    found = False
    for i, s in sigs:
        idx = i
        if s == '-':
            ns = '+'
        elif s == '+':
            ns = '-'
        new_fact = list(final_fact)
        new_fact[idx] = ns
        new_fact = "".join(new_fact)
        
    # elif s == '*':
    #     ns = '/'
    # elif s == '/':
    #     ns = '*'
    if not found:
        return state
    print("init", final_fact)
    final_fact = list(final_fact)
    final_fact[idx] = ns
    final_fact = "".join(final_fact)
    print("final", final_fact)
    facts = list(state.facts)
    facts[-1] = final_fact
    state.facts = tuple(facts)
    return state
                    

def rollout(model,
            environment,
            state,
            max_steps,
            beam_size=1,
            corrupt=0.,
            debug=False):
    """Runs beam search using the Q value until either
    max_steps have been made or reached a terminal state."""
    beam = [state]
    history = [beam]
    seen = set([state])
    success = False
    corrupt = False

    for i in range(max_steps):
        if debug:
            print(f'Beam #{i}: {beam}')

        if not beam:
            break

        rewards, s_actions = zip(*environment.step(beam))
        actions = [a for s_a in s_actions for a in s_a]

        if max(rewards):
            success = True
            break

        if len(actions) == 0:
            success = False
            break

        with torch.no_grad():
            q_values = model(actions).tolist()

        for a, v in zip(actions, q_values):
            a.next_state.value = model.aggregate(a.state.value, v)

        ns = list(set([a.next_state for a in actions]) - seen)
        ns.sort(key=lambda s: s.value, reverse=True)
        if random.uniform(0, 1) < corrupt:
            ns = [corrupt_state(s) for s in ns] 
            corrupt = True
        if debug:
            print(f'Candidates: {[(s, s.value) for s in ns]}')
        beam = ns[:beam_size]
        history.append(ns)
        seen.update(ns)
    if corrupt:
        success = False
    return success, history




def get_algebra_data(num_states=None):
    train_dataset = load_dataset('json', is_train=True)
    if num_states is None:
        num_states = train_dataset.n_problems
    problem_states = [environment.State([p], [], 0) for p in train_dataset.problems[:num_states]]
    return problem_states


def evaluate_solver(problems, checkpoint, beam_size, max_steps, corrupt=0., debug=False):
    model = torch.load(checkpoint, map_location=device)
    model.to(device)
    env = environment.RustEnvironment("equations-ct")
    responses = []
    histories = []
    scores = 0.
    total = 0.
    pbar = tqdm(problems)

    for state in pbar:
        total += 1
        success, history = rollout(model, env, state,
                                         max_steps, beam_size, corrupt, debug)
        histories.append(history)
        if success:
            scores += 1
            responses.append(1)
        else:
            responses.append(0)
        pbar.set_description(f"current score {scores / total}")
    return responses, histories


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--population-type', type=str, default='epoch',
                        choices=['epoch', 'depth', 'beam'],
                        help='epoch|depth|beam (default: epoch)')
    parser.add_argument('--ckpt-path', type=str, default='/mnt/fs3/poesia/socratic-tutor/output/algebra-solver/ConPoLe/equations-ct/run0/checkpoints/',
                        help='path to checkpoints')
    parser.add_argument('--save-path', type=str, default='/mnt/fs1/kanishkg/rich-irt/variational-item-response-theory-public/data/algebra',
                        help='path where to save the results')
    parser.add_argument('--save-file', type=str, default='algebra_steps',
                        help='name of the file to save the results')
    parser.add_argument('--max-depth', type=int, default=30,
                        help='maximum depth of search')
    parser.add_argument('--beam-size', type=int, default=10,
                        help='size of beam search')
    parser.add_argument('--corrupt', type=float, default=0.,
                        help='probability of corrupting states')
    parser.add_argument('--best-epoch', type=int, default=88,
                        help='number of the best epoch')
    parser.add_argument('--num-states', type=int, default=None,
                        help='number of problems to evaluate')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='whether to print debug info like steps')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='whether to use cuda')    
    parser.add_argument('--min', type=int, default=0,
                        help='what is the minimun amount of parameters')
    parser.add_argument('--max', type=int, default=0,
                        help='what is the maximum amount of parameters')



    args = parser.parse_args()
    min_param = args.min
    max_param = args.max

    if args.population_type == 'beam-size':
        population_parameters = {'beam-size': reversed([i + 1 for i in range(min_param, max_param)])}
    elif args.population_type == 'epoch':
        best_epoch = 88
        population_parameters = {'epoch': [args.best_epoch - i for i in range(min_param, max_param)]}
    elif args.population_type == 'depth':
        population_parameters = {'depth': [i + 1 for i in range(min_param, max_param)]}

    device = torch.device("cuda" if args.cuda else "cpu")

    train_dataset = load_dataset('json', is_train=True)
    problem_states = get_algebra_data(args.num_states)
    responses = []
    for p in population_parameters[args.population_type]:
        dataset = {'response': [], 'epoch': [], 'beam': [], 'depth': [], 'score':[], 'steps':[],
                   'problems': train_dataset.problems}
        epoch = args.best_epoch
        depth = args.max_depth
        beam = args.beam_size
        if args.population_type == 'beam-size':
            res, steps = evaluate_solver(problem_states, os.path.join(args.ckpt_path, f'{args.best_epoch}.pt'), p, args.max_depth, args.corrupt, args.debug)
            beam = p
        elif args.population_type == 'epoch':
            res, steps = evaluate_solver(problem_states, os.path.join(args.ckpt_path, f'{p}.pt'), args.beam_size, args.max_depth, args.corrupt, args.debug)
            epoch = p
        elif args.population_type == 'depth':
            res, steps = evaluate_solver(problem_states, os.path.join(args.ckpt_path, f'{args.best_epoch}.pt'), args.beam_size, p, args.corrupt, args.debug)
            depth = p

        dataset['response'].append(res)
        dataset['epoch'].append(epoch)
        dataset['beam'].append(beam)
        dataset['depth'].append(depth)
        dataset['score'].append(sum(res)/len(res))
        dataset['steps'].append(steps)
        print(f"epoch: {epoch}, beam: {beam}, depth: {depth}, score: {sum(res)/len(res)}, corrupt: {args.corrupt}")
        torch.save(dataset, os.path.join(args.save_path,
                                     f'{args.save_file}_{epoch}_{beam}_{depth}_{args.corrupt}.pth'))
