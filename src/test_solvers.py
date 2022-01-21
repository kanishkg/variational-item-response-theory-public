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


def evaluate_solver(problems, checkpoint, beam_size, max_steps):
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
                                         max_steps, beam_size, debug)
        if success:
            scores += 1
            responses.append(1)
        else:
            responses.append(0)
        pbar.set_description(f"current score {scores / total}")
    return responses


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    cuda = True
    max_steps = 40  # Maximum length of an episode.
    beam_size = 60  # Size of the beam in beam search.
    debug = False  # Whether to print all steps during evaluation.
    ckpt_path = '/mnt/fs3/poesia/socratic-tutor/output/algebra-solver/ConPoLe/equations-ct/run0/checkpoints/'
    beam_size = 3
    max_depth = 50
    ckpt = 88
    population_type = "beam-size"
    num_states = None

    if population_type == 'beam-size':
        population_parameters = {'beam-size': reversed([i + 1 for i in range(20)])}
    elif population_type == 'epoch':
        best_epoch = 88
        population_parameters = {'epoch': [best_epoch - i for i in range(40)]}
    elif population_type == 'depth':
        population_parameters = {'epoch': [i + 1 for i in range(70)]}

    device = torch.device("cuda" if cuda else "cpu")

    train_dataset = load_dataset('json', is_train=True)
    problem_states = get_algebra_data(num_states)
    responses = []
    dataset = {'response': [], 'epoch': [], 'beam': [], 'depth': [], 'score':[],
                   'problems': train_dataset.problems}

    for p in population_parameters[population_type]:
        epoch = ckpt
        depth = max_depth
        beam = beam_size
        if population_type == 'beam-size':
            res = evaluate_solver(problem_states, os.path.join(ckpt_path, f'{ckpt}.pt'), p, max_depth)
            beam = p
        elif population_type == 'epoch':
            res = evaluate_solver(problem_states, os.path.join(ckpt_path, f'{p}.pt'), beam_size, max_depth)
            epoch = p
        elif population_type == 'depth':
            res = evaluate_solver(problem_states, os.path.join(ckpt_path, f'{ckpt}.pt'), beam_size, p)
            depth = p
        dataset['response'].append(res)
        dataset['epoch'].append(epoch)
        dataset['beam'].append(beam)
        dataset['depth'].append(depth)
        dataset['score'].append(sum(res)/len(res))
        print(f"epoch: {epoch}, beam: {beam}, depth: {depth}, score: {sum(res)/len(res)}")
        torch.save(dataset, os.path.join('/mnt/fs1/kanishkg/rich-irt/variational-item-response-theory-public/data/algebra',
                                     'algebra2.pth'))
