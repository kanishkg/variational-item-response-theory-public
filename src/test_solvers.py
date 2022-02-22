import os
import re
import time
import random
import math
import numpy as np
from tqdm import tqdm
import sys
import subprocess
import copy
import torch

from src.datasets import load_dataset, artificially_mask_dataset, collate_function_step
import environment
import argparse

sys.path.append('../../socratic-tutor/')

signs = ['+', '-']
symbols = ['(', ')', ' ']


def filter_problem(problem):
    # remove parsing error of x in denominator
    xids = [i for i, c in enumerate(problem) if c == 'x']
    for _ in range(len(xids)):
        nums = re.findall('[0-9]+', problem)
        ids = [m.start(0) for m in re.finditer('[0-9]+', problem)]
        # ids = [problem.index(n) for n in nums]
        for j, (idx, num) in enumerate(zip(ids, nums)): 
            if idx+len(num) >= len(problem):
                continue
            if problem[idx+len(num)] == 'x':
                if problem[idx-2] == '/':
                    prev_num = [ids[j-1], nums[j-1]]
                    if problem[prev_num[0]-1] != '-':
                        problem = problem[:prev_num[0]]+'('+problem[prev_num[0]:idx+len(num)] + ') * '  + problem[idx+len(num):]
                    else:
                        problem = problem[:prev_num[0]-1]+'('+problem[prev_num[0]-1:idx+len(num)] + ') * '  + problem[idx+len(num):]
                    break
    return problem    

def filter_state(state):
    fact = state.facts[-1]
    if  '+ (+' in fact:
        fact = fact.replace('+ (+', '+ (')
    if '- +' in fact:
        fact = fact = fact.replace('- +', '- ')
    if '(+' in fact:
        fact = fact.replace('(+','(')
    if '+ +' in fact:
        fact = fact.replace('+ +', '+ ')
    if '[+' in fact:
        fact = fact.replace('[+', '[')
    if '- +' in fact:
        fact = fact.replace('- +', '- ')
    if fact[0] == '+':
        fact = fact[1:]
    if '* +' in fact:
        fact = fact.replace('* +', '* ')
    if '= +' in fact:
        fact = fact.replace('= +', '= ')
    if '/ +' in fact:
        fact = fact.replace('/ +', '/ ')
    facts = list(state.facts)
    facts[-1] = fact
    state.facts = tuple(facts)
    # if fact!=init_fact:
    #     print(f'{init_fact} -> {fact}')
    return state

def corrupt_vars(fact):
    init_fact = fact
    # choose to add or delete variables
    prob = random.uniform(0, 1)
    # randomly delete a var from the equation
    if fact.count('x') == 1:
        prob = 1.
    if prob < 0.5:
        # get ids of characters in fact
        ids = [i for i, c in enumerate(fact) if c == 'x']
        # randomly choose an id and delete variables
        idx = random.choice(ids)
        fact = fact[:idx]+fact[idx+1:]
    else:
        # randomly add a variable to the equation
        nums = re.findall('[0-9]+', fact)
        # get start and end positions of numbers
        ids = [(m.start(0), m.end(0)) for m in re.finditer('[0-9]+', fact)]

        ids_nums = list(zip(ids, nums))
        random.shuffle(ids_nums)
        for i, n in ids_nums:
            if fact[i[1]] == 'x':
                continue
            if fact[i[1]] == ']':
                continue
            if fact[i[0]-1] == '[':
                continue
            if fact[i[0]-1] == '/':
                continue
            fact = fact[:i[1]]+'x'+fact[i[1]:]
            break
    return fact

def corrupt_sigs(fact):
    sigs = [(i, s) for i, s in enumerate(fact) if s in signs]
    random.shuffle(sigs)
    idx, s = sigs[0]
    if s == '-':
        ns = '+'
    elif s == '+':
        ns = '-'
    fact = list(fact)
    fact[idx] = ns
    fact = "".join(fact)
    return fact


def corrupt_state(state):
    final_fact = state.facts[-1]
    success = True
    # choose how to corrupt the equation
    p = random.uniform(0, 1)
    sigs = [(i, s) for i, s in enumerate(final_fact) if s in signs]
    if len(sigs) == 0:
        p = 1.
    if p < 0.5:
        final_fact = corrupt_sigs(final_fact)
    else:
        final_fact = corrupt_vars(final_fact)
    if final_fact == state.facts[-1]:
        success = False
    facts = list(state.facts)
    facts[-1] = final_fact
    state.facts = tuple(facts)
    return state, success
                    

def rollout(model,
            env,
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
    is_corrupt = False

    for i in range(max_steps):
        if debug:
            print(f'Beam #{i}: {beam}')

        if not beam:
            break

        rewards, s_actions = zip(*env.step(beam))
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
            corruption_results = [corrupt_state(s) for s in ns]
            ns = [s for s, _ in corruption_results] 
            corruption = [s for _, s in corruption_results] 
            # TODO: if corruption is not successful, we don't want to change the response
            is_corrupt = True 
        ns = [filter_state(s) for s in ns]
        # ns = [environment.State([f], [], 0) for f in fin_fact]
        if debug:
            print(f'Candidates: {[(s, s.value) for s in ns]}')
        beam = ns[:beam_size]
        history.append(ns)
        seen.update(ns)
    answer_facts = history[-1][0].facts
    if is_corrupt:
        success = False
    
    return success, history




def get_algebra_data(num_states=None):
    train_dataset = load_dataset('json', is_train=True)
    if num_states is None:
        num_states = train_dataset.n_problems
    problem_states = []
    changed = 0
    f = open('changed_problems.txt', 'w')
    for p in train_dataset.problems[:num_states]:
        init_p = copy.deepcopy(p)
        p = filter_problem(p)
        if init_p != p:
            f.write(f"{init_p} -> {p}\n")
            changed +=1
        problem_states += [environment.State([filter_problem(p)], [], 0)]
    print(f'Changed {changed} out of {num_states}')
    f.close()
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
        histories.append(history[-1])
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
                        choices=['epoch', 'depth', 'beam', 'corrupt'],
                        help='epoch|depth|beam|corrupt (default: epoch)')
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
    elif args.population_type == 'corrupt':
        population_parameters = {'corrupt': reversed([i * 1e-3 for i in range(min_param, max_param)])}


    device = torch.device("cuda" if args.cuda else "cpu")

    train_dataset = load_dataset('json', is_train=True)
    problem_states = get_algebra_data(args.num_states)
    responses = []
    for p in population_parameters[args.population_type]:
        dataset = {'response': [], 'epoch': [], 'beam': [], 'depth': [], 'score':[], 'steps':[], 'corrupt':[],
                   'problems': train_dataset.problems}
        epoch = args.best_epoch
        depth = args.max_depth
        beam = args.beam_size
        corrupt = args.corrupt
        if args.population_type == 'beam-size':
            res, steps = evaluate_solver(problem_states, os.path.join(args.ckpt_path, f'{args.best_epoch}.pt'), p, args.max_depth, args.corrupt, args.debug)
            beam = p
        elif args.population_type == 'epoch':
            res, steps = evaluate_solver(problem_states, os.path.join(args.ckpt_path, f'{p}.pt'), args.beam_size, args.max_depth, args.corrupt, args.debug)
            epoch = p
        elif args.population_type == 'depth':
            res, steps = evaluate_solver(problem_states, os.path.join(args.ckpt_path, f'{args.best_epoch}.pt'), args.beam_size, p, args.corrupt, args.debug)
            depth = p
        elif args.population_type == 'corrupt':
            res, steps = evaluate_solver(problem_states, os.path.join(args.ckpt_path, f'{args.best_epoch}.pt'), args.beam_size, args.max_depth, p, args.debug)
            corrupt = p
        dataset['response'].append(res)
        dataset['epoch'].append(epoch)
        dataset['beam'].append(beam)
        dataset['depth'].append(depth)
        dataset['score'].append(sum(res)/len(res))
        dataset['steps'].append(steps)
        dataset['corrupt'].append(corrupt)
        print(f"epoch: {epoch}, beam: {beam}, depth: {depth}, score: {sum(res)/len(res)}, corrupt: {corrupt}")
        torch.save(dataset, os.path.join(args.save_path,
                                     f'{args.save_file}_{epoch}_{beam}_{depth}_{corrupt:.3f}.pth'))
