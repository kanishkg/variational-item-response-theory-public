import copy
import os
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
            predicted_move = engine.get_best_move()
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
    stochfish_path = "/mnt/fs6/kanishkg/Stockfish/src/stockfish"
    engine_name = 'stockfish'
    population_type = 'level'
    num_puzzles = 100

    if engine_name == 'stockfish':
        engine = Stockfish(path=stochfish_path)
        if population_type == 'level':
            population_parameters = {'level': [i + 1 for i in range(20)]}

    data = get_chess_data(data_file)
    responses = []
    ability = []
    item_difficulty = [d['Rating'] for d in data[:num_puzzles]]
    for p in population_parameters[population_type]:
        engine.set_skill_level(p)
        responses.append(test_engine(engine, data, num_puzzles))
        ability.append(p)
        print(p, sum(responses)/len(responses))
    dataset = {'response': responses, 'ability': ability, 'item_feat': item_difficulty}

    torch.save(dataset, os.path.join('/mnt/fs1/kanishkg/rich-irt/variational-item-response-theory-public/data/chess', 'chess.pth'))