import copy
import os
from subprocess import Popen, PIPE, run

from tqdm import tqdm

import torch
import chess
from stockfish import Stockfish

def get_chess_data(data_file):
    # PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,Themes,GameUrl
    puzzles = []
    with open(data_file, 'r') as f:
        raw_problems = f.read().split('\n')[:-1]
    for problem in tqdm(raw_problems):
        puzzle = {}
        problem_list = problem.split(',')
        puzzle['PuzzleId'] = problem_list[0]
        puzzle['FEN'] = problem_list[1]
        puzzle['Moves'] = problem_list[2].split()
        puzzle['Rating'] = problem_list[3]
        puzzle['RatingDeviation'] = problem_list[4]
        puzzle['Popularity'] = problem_list[5]
        puzzle['NbPlays'] = problem_list[6]
        puzzle['Themes'] = problem_list[7]
        puzzle['GameUrl'] = problem_list[8]
        puzzles.append(puzzle)
    print(f"found {len(puzzles)}")
    return puzzles

class Leela(object):
    def __init__(self):
        super().__init__()
        self.fen = None
        self.weights = None
        self.nodes = None

    def set_skill_level(self, weights, nodes=-1):
        self.nodes = nodes
        self.weights = weights
        command = ['lc0', 'describenet', f'--weights={self.weights}']
        result = run(command, stdout=PIPE, stderr=PIPE, text=True)
        assert result.returncode == 0
        out = result.stdout.split()
        self.elo = int(weights.split('_')[-1])
        self.train_steps = out[out.index('steps:')+1]
        self.policy_loss = out[out.index('Policy')+2]
        self.mse_loss = out[out.index('MSE')+2]
        self.accuracy = out[out.index('Accuracy:')+1]

    def get_parameters(self):
        return f"elo: {self.elo}, train_steps: {self.train_steps}, nodes: {self.nodes}, accuracy = {self.accuracy}," \
               f" losses: {self.mse_loss,self.policy_loss}"

    def set_fen_position(self, fen):
        self.fen = fen

    def get_best_move_time(self, time):
        assert self.fen is not None
        assert self.weights is not None
        assert self.nodes is not None

        command = ['lc0', 'benchmark', f'--weights={self.weights}',
                   f'--fen={self.fen}', f'--movetime={time}', f'--nodes={self.nodes}']
        result = run(command, stdout=PIPE, stderr=PIPE, text=True)
        assert result.returncode == 0
        out = result.stdout.split()
        best_move = out[out.index('bestmove')+1]
        return best_move


def test_engine(engine, data, num_puzzles=-1):
    responses = []
    scores = 0.
    total = 0.
    pbar = tqdm(data[:num_puzzles])

    for p in pbar:
        total +=1
        fen = p['FEN']
        moves = p['Moves']
        board = chess.Board(fen)
        opponent, query = moves[::2], moves[1::2]
        score = 1

        for q in query:
            opp_move = opponent[0]
            del (opponent[0])
            board.push(chess.Move.from_uci(opp_move))
            fen = board.fen()
            engine.set_fen_position(fen)
            predicted_move = engine.get_best_move_time(100)
            if predicted_move != q:
                board_test = copy.deepcopy(board)
                board_test.push(chess.Move.from_uci(predicted_move))
                if not board_test.is_checkmate():
                    score = 0
                    break
            board.push(chess.Move.from_uci(q))
        responses.append(score)
        scores += score
        pbar.set_description(f"current score {scores/total}")
    return responses


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    data_file = '/mnt/fs1/kanishkg/rich-irt/variational-item-response-theory-public/data/chess/lichess_db_puzzle.csv'
    engine_name = 'leela'

    engine_path_stockfish = "/mnt/fs6/kanishkg/Stockfish/src/stockfish"
    engine_path_leela = "/mnt/fs6/kanishkg/lc0/weights"

    population_type = 'level'
    nodes = -1
    num_puzzles = 1000

    if engine_name == 'stockfish':
        engine = Stockfish(path=engine_path_stockfish)
        if population_type == 'level':
            population_parameters = {'level': [i + 1 for i in range(20)]}

    elif engine_name == 'leela':
        engine = Leela()
        if population_type == 'level':
            weight_files = os.listdir(engine_path_leela)
            population_parameters = {'level': sorted([os.path.join(engine_path_leela, w) for w in weight_files])}

    data = get_chess_data(data_file)
    responses = []
    ability = []
    item_difficulty = [d['Rating'] for d in data[:num_puzzles]]

    dataset = {'response': [], 'train_steps': [], 'accuracy': [], 'policy_loss': [], 'mse_loss':[], 'elo': [],
               'item_feat': item_difficulty}
    for p in population_parameters[population_type]:
        engine.set_skill_level(p)
        res = test_engine(engine, data, num_puzzles)
        dataset['response'].append(res)
        print(engine.get_parameters())
        if engine_name == 'leela':
            dataset['elo'].append(engine.elo)
            dataset['nodes'].append(engine.nodes)
            dataset['accuracy'].append(engine.accuracy)
            dataset['train_steps'].append(engine.train_steps)
            dataset['policy_loss'].append(engine.policy_loss)
            dataset['mse_loss'].append(engine.mse_loss)

        torch.save(dataset, os.path.join('/mnt/fs1/kanishkg/rich-irt/variational-item-response-theory-public/data/chess',
                                     'leela.pth'))