import argparse
import random
import sys
import time

import torch
import numpy as np

from board import Board
from encoding import ACTION_TABLE, encode_state, get_legal_mask, decode_action
from env import make_initial_state
from train import Agent
from scipy.stats import binomtest
from rules import advance_turn, get_legal_moves


NAMES      = ["Alice (PPO)", "Bob (rand)", "Carol (rand)"]
CARD_NAMES = {1:"A",2:"2",3:"3",4:"4",5:"5",6:"6",7:"7",
              8:"8",9:"9",10:"10",11:"J",12:"Q",13:"K"}

SEGMENT    = 18
HOME_START = {0: 54, 1: 58, 2: 62}
HOME_END   = {0: 57, 1: 61, 2: 65}
JUST_OUT   = {0: 0,  1: 18, 2: 36}
HOME_ENTRY = {0: 52, 1: 16, 2: 34}


def run_game(agent: Agent, board: Board, device: torch.device, max_turns: int = 1000, verbose: bool = True, seed: int | None = None) -> int:
    if seed is not None:
        random.seed(seed)

    state = make_initial_state()
    turn  = 0


    while turn < max_turns:
        active = state.active_player

        if active == 0:
            # PPO agent
            obs  = torch.as_tensor(encode_state(state), dtype=torch.float32).unsqueeze(0).to(device)
            mask_list = get_legal_mask(state, board, ACTION_TABLE)
            mask = torch.tensor(mask_list, dtype=torch.bool).unsqueeze(0).to(device)
            with torch.no_grad():
                action_idx, _, _, _ = agent.get_action_and_value(obs, mask)
            action = decode_action(ACTION_TABLE, action_idx.item(), state, board)
        else:
            # Random agent
            legal  = get_legal_moves(state, board)
            action = random.choice(legal)

        state, game_over = advance_turn(state, action)
        turn += 1

        if game_over:
            winner = state.active_player
            if verbose:
                label = f"{NAMES[winner]} wins" if winner == 0 else f"{NAMES[winner]} wins"
                print(f"{label} in {turn} turns!")
            return winner

    if verbose:
        print(f"Game did not finish within {max_turns} turns.")
    return -1


def benchmark(agent, board, device, n_games, seed_offset=0):
    wins = [0, 0, 0]
    unfinished = 0
    lengths = []

    for i in range(n_games):
        result = run_game(agent, board, device,
                          delay=0, verbose=False, seed=seed_offset + i,
                          max_turns=1000)
        if result == -1:
            unfinished += 1
        else:
            wins[result] += 1

    print(f"=== Benchmark: {n_games} games ===")
    for p in range(3):
        pct = 100 * wins[p] / n_games
        bar = "█" * int(pct / 2)
        print(f"  {NAMES[p]:30s}  wins: {wins[p]:4d} / {n_games}  ({pct:5.1f}%)  {bar}")
    if unfinished:
        print(f"  Unfinished: {unfinished}")

    print(f"\n  Random baseline win rate: ~33.3%")
    ppo_pct = 100 * wins[0] / n_games
    delta = ppo_pct - 33.3
    sign  = "+" if delta >= 0 else ""
    print(f"  PPO vs baseline: {sign}{delta:.1f}pp\n")


    n_games = n_games
    n_wins = wins[0]

    result = binomtest(n_wins, n=n_games, p=1 / 3, alternative='greater')
    print(f"p-value:  {result.pvalue:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to checkpoint .pt file")
    parser.add_argument("--games",  type=int,   default=1)
    parser.add_argument("--seed",   type=int,   default=None)
    parser.add_argument("--device", type=str,   default="cpu",
                        help="cpu | cuda | mps")
    args = parser.parse_args()

    device = torch.device(args.device)

    agent = Agent().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt["agent"] if isinstance(ckpt, dict) else ckpt
    agent.load_state_dict(state_dict)

    agent.eval()

    board = Board()

    if args.games == 1:
        seed = args.seed if args.seed is not None else random.randint(0, 9999)
        run_game(agent, board, device, verbose=True, seed=seed)
    else:
        seed_offset = args.seed if args.seed is not None else random.randint(0, 9999)
        benchmark(agent, board, device, n_games=args.games, seed_offset=seed_offset)


if __name__ == "__main__":
    main()