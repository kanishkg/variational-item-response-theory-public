import copy
from tqdm import tqdm

import chess
from stockfish import Stockfish


def get_chess_data(data_file):
    # PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,Themes,GameUrl
    puzzles = []
    with open(data_file, 'r') as f:
        raw_problems = f.read().split('\n')
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


def test_engine(engine, data):
    responses = []
    for p in tqdm(data):
        fen = p['FEN']
        moves = p['moves']
        board = chess.Board(fen)
        opponent, query = moves[::2], moves[1::2]
        score = True

        for q in query:
            opp_move = opponent[0]
            del (opponent[0])
            board.push(chess.Move.from_uci(opp_move))
            fen = board.fen()
            engine.set_fen_position(fen)
            predicted_move = engine.get_best_move()
            print(predicted_move)
            if predicted_move != q:
                board_test = copy.deepcopy(board)
                board_test.push(chess.Move.from_uci(predicted_move))
                if not board_test.is_checkmate():
                    score = False
                    break
            board.push(chess.Move.from_uci(q))
        responses.append(score)
    return responses


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    data_file = '/mnt/fs1/kanishkg/rich-irt/variational-item-response-theory-public/data/chess/lichess_db_puzzle.csv'
    stochfish_path = "/mnt/fs6/kanishkg/Stockfish/src/stockfish"
    engine_name = 'stockfish'
    population_type = 'level'

    if engine_name == 'stockfish':
        engine = Stockfish(path=stochfish_path)
        if population_type == 'level':
            population_parameters = {'level': [i + 1 for i in range(20)]}

    data = get_chess_data(data_file)
    for p in population_parameters[population_type]:
        engine.set_level(1)
