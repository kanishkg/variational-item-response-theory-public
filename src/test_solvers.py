import os
import time
import math
import numpy as np
from tqdm import tqdm
import sys

import torch

from src.datasets import load_dataset, artificially_mask_dataset, collate_function_step
import environment

sys.path.append('../../socratic-tutor/')

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()

    train_dataset = load_dataset(
        'json',
        is_train = True,
    )
    cuda = True
    max_steps = 40  # Maximum length of an episode.
    beam_size = 60  # Size of the beam in beam search.
    debug = False# Whether to print all steps during evaluation.

    ckpt_path = '/mnt/fs3/poesia/aws-output/a1b2c/NCE+H/equations-ct/run0/checkpoints/'
    ckpt = 100

    device = torch.device("cuda" if cuda else "cpu")
    def evaluate_solver(checkpoint, dataset):
        model = torch.load(checkpoint, map_location=device)
        model.to(device)
        env = environment.RustEnvironment("equations-ct")
        n_problems = train_dataset.n_problems  # How many problems to use.

        successes = []
        solution_lengths = []
        failures = []
        states = [environment.State([train_dataset.problems[i]], [], 0) for i in range(dataset.n_problems)]

        for i, state in enumerate(states):
            success, history = model.rollout(env, state,
                                         max_steps, beam_size, debug)

            if success:
                successes.append((i, train_dataset.problems[i]))
            else:
                failures.append((i, train_dataset.problems[i]))
            solution_lengths.append(len(history) - 1 if success else -1)
            print(i, train_dataset.problems[i], '-- success?', success)

        return {
            'success_rate': len(successes) / n_problems,
            'solution_lengths': solution_lengths,
            'max_solution_length': max(solution_lengths),
            'successes': successes,
            'failures': failures,
            'beam': beam_size,
            'max_steps': max_steps
        }

    results = evaluate_solver(os.path.join(ckpt_path, f'{ckpt}.pt'), train_dataset)
    print(results)

