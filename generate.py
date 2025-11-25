import chess
import chess.engine
import multiprocessing
import random
import struct
from pathlib import Path
import sys
import numpy as np
sys.path.append(str(Path(__file__).parent / 'GoodKnightCommon'))
from fen_to_tensor import get_tensor_bytes_from_fen, NUM_BYTES

def normalize_centipawns(centipawns):
    """
    Normalize centipawn evaluation using tanh function.
    Maps centipawn values to range [-1, 1] where:
    - 0 = even position (0 centipawns)
    - +1 = white winning
    - -1 = black winning

    Uses scaling factor to control steepness of tanh.
    """
    # Scale factor: +-400 centipawns maps to ~+-0.76
    # +-800 centipawns maps to ~+-0.93
    scale = 400.0
    return np.tanh(centipawns / scale)

def worker(stop_flag, proc_id, stockfish_path):
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    output_dir = Path('data')
    output_dir.mkdir(exist_ok=True)

    game_count = 0
    positions = []

    try:
        while not stop_flag.value:
            board = chess.Board()
            move_count = 0

            # Play one game
            while not board.is_game_over():
                move_count += 1
                # Get top 5 moves with evaluations
                info = engine.analyse(board, chess.engine.Limit(time=0.1, depth=10), multipv=5)

                # Store position data (FEN, evaluation, best move)
                best_eval = info[0]["score"].relative.score(mate_score=10000)
                best_move = info[0]["pv"][0]

                # don't want a bunch of the starting moves inflating the size
                # of the training data.
                if move_count >= 5:
                    # Convert FEN to bytes using GoodKnightCommon
                    fen_bytes = get_tensor_bytes_from_fen(board.fen())
                    positions.append({
                        'fen_bytes': fen_bytes,
                        'eval': best_eval
                    })

                # Pick a move from top 5 with some randomness
                move = random.choice([pv["pv"][0] for pv in info])
                board.push(move)

            game_count += 1

            # Write to file when we hit chunk size
            if len(positions) >= 10000:
                write_positions_to_file(positions, output_dir, proc_id, game_count)
                positions = []

            if game_count % 10 == 0:
                print(f"[Worker {proc_id}] Generated {game_count} games, {len(positions)} positions in buffer")

    except KeyboardInterrupt:
        pass
    finally:
        # Write any remaining positions
        if positions:
            write_positions_to_file(positions, output_dir, proc_id, game_count)
        engine.quit()
        print(f"[Worker {proc_id}] Shutting down. Total games: {game_count}")


def write_positions_to_file(positions, output_dir, proc_id, game_count):
    """
    Write positions to .npz file with separate input and output arrays.

    Input: board positions as tensors (N, 18, 8, 8) - uint8
    Output: normalized evaluations (N,) - float32 in range [-1, 1]
    """
    filename = output_dir / f"worker_{proc_id}_chunk_{game_count}.npz"

    # Convert fen_bytes to numpy arrays
    input_data = []
    output_data = []

    for pos in positions:
        # Convert bytes to numpy array and reshape to (18, 8, 8)
        # Each position is 18 channels * 8 rows * 8 cols = 1152 bytes
        board_tensor = np.frombuffer(pos['fen_bytes'], dtype=np.uint8).reshape(18, 8, 8)
        input_data.append(board_tensor)

        # Normalize the centipawn evaluation using tanh
        normalized_eval = normalize_centipawns(pos['eval'])
        output_data.append(normalized_eval)

    # Convert lists to numpy arrays
    input_array = np.array(input_data, dtype=np.uint8)
    output_array = np.array(output_data, dtype=np.float32)

    # Save as compressed npz file
    np.savez_compressed(
        filename,
        input=input_array,
        output=output_array
    )

    print(f"[Worker {proc_id}] Wrote {len(positions)} positions to {filename}")


def main():
    # find the stockfish executable path
    stockfish_path = Path('.') / 'stockfish' / 'stockfish-ubuntu-x86-64'
    if not stockfish_path.exists():
        print(f"[ERROR] Couldn't find stockfish at path {stockfish_path} try running get_stockfish.sh")
        return

    processes = []
    manager = multiprocessing.Manager()
    stop_flag = manager.Value('i', 0)

    try:
        for proc_id in range(multiprocessing.cpu_count()):
            process = multiprocessing.Process(target=worker, args=(stop_flag, proc_id, stockfish_path,))
            process.start()
            processes.append(process)

        # Wait for processes
        for process in processes:
            process.join()

    except KeyboardInterrupt:
        print("\n[Main] Stopping workers...")
        stop_flag.value = 1
        for process in processes:
            process.join()


if __name__ == '__main__':
    main()
