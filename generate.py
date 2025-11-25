import chess
import chess.engine
import multiprocessing
import random
import struct
from pathlib import Path
import sys
import numpy as np
import signal
import time
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
    engine = None
    try:
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        print(f"[Worker {proc_id}] Started successfully")
    except Exception as e:
        print(f"[Worker {proc_id}] Failed to start engine: {e}")
        return

    output_dir = Path('data')
    output_dir.mkdir(exist_ok=True)

    game_count = 0
    positions = []
    total_positions = 0
    start_time = time.time()

    try:
        while not stop_flag.value:
            board = chess.Board()
            move_count = 0

            # Play one game
            while not board.is_game_over() and not stop_flag.value:
                move_count += 1

                try:
                    # Get top 5 moves with evaluations
                    info = engine.analyse(board, chess.engine.Limit(time=0.1, depth=10), multipv=5)
                except chess.engine.EngineTerminatedError:
                    print(f"[Worker {proc_id}] Engine terminated unexpectedly")
                    return

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
                total_positions += len(positions)
                positions = []

                # More detailed progress logging
                elapsed = time.time() - start_time
                rate = total_positions / elapsed if elapsed > 0 else 0
                print(f"[Worker {proc_id}] Progress: {game_count} games | {total_positions} positions | {rate:.1f} pos/sec")

            if game_count % 10 == 0:
                print(f"[Worker {proc_id}] Generated {game_count} games, {len(positions)} positions in buffer")

    except KeyboardInterrupt:
        print(f"[Worker {proc_id}] Received interrupt signal")
    except Exception as e:
        print(f"[Worker {proc_id}] Error during generation: {e}")
    finally:
        # Write any remaining positions
        if positions:
            print(f"[Worker {proc_id}] Writing final {len(positions)} positions...")
            write_positions_to_file(positions, output_dir, proc_id, game_count)
            total_positions += len(positions)

        # Gracefully quit engine with timeout
        if engine:
            try:
                engine.quit()
            except Exception as e:
                print(f"[Worker {proc_id}] Error quitting engine: {e}")

        elapsed = time.time() - start_time
        print(f"[Worker {proc_id}] Shutdown complete. Total: {game_count} games, {total_positions} positions in {elapsed:.1f}s")


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

    num_workers = multiprocessing.cpu_count()
    print(f"[Main] Starting {num_workers} worker processes...")

    processes = []
    manager = multiprocessing.Manager()
    stop_flag = manager.Value('i', 0)

    # Ignore SIGINT in main process - we'll handle it manually
    original_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)

    try:
        for proc_id in range(num_workers):
            process = multiprocessing.Process(target=worker, args=(stop_flag, proc_id, stockfish_path,))
            process.start()
            processes.append(process)

        # Restore SIGINT handler
        signal.signal(signal.SIGINT, original_sigint)

        print(f"[Main] All workers started. Press Ctrl+C to stop.\n")

        # Wait for processes
        for process in processes:
            process.join()

    except KeyboardInterrupt:
        print("\n[Main] Interrupt received. Stopping workers gracefully...")
        stop_flag.value = 1

        # Give workers time to finish current operations
        print("[Main] Waiting for workers to finish (max 30 seconds)...")
        for i, process in enumerate(processes):
            try:
                process.join(timeout=30)
                if process.is_alive():
                    print(f"[Main] Worker {i} did not finish in time, terminating...")
                    process.terminate()
                    process.join(timeout=5)
            except Exception as e:
                print(f"[Main] Error stopping worker {i}: {e}")

        print("[Main] All workers stopped.")


if __name__ == '__main__':
    main()
