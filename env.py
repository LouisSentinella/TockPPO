import random

import gymnasium as gym
import numpy as np

from board import Board
from encoding import encode_state, decode_action, get_legal_mask, ACTION_TABLE
from rules import advance_turn
from state import GameState, Pawn, Zone


def make_initial_state(deal_starting_player: int = 0) -> GameState:
    deck = list(range(1, 14)) * 4
    random.shuffle(deck)

    deal_size = 5
    hands = []
    for _ in range(3):
        hands.append(deck[:deal_size])
        deck = deck[deal_size:]

    pawns = [
        [Pawn(owner=p, pawn_id=i, zone=Zone.BASE, index=None) for i in range(4)]
        for p in range(3)
    ]

    return GameState(
        pawns=pawns,
        hands=hands,
        deck=deck,
        discard_pile=[],
        active_player=deal_starting_player,
        skip_flag=False,
        deal_round=1,
        deal_starting_player=deal_starting_player,
    )


CAPTURE_REWARD = 0.1
HOME_REWARD    = 0.1


class TockEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.board = Board()
        self.action_table = ACTION_TABLE
        self.observation_space = gym.spaces.Box(
            low=-1, high=65, shape=(30,), dtype=np.int64
        )
        self.action_space = gym.spaces.Discrete(len(ACTION_TABLE))
        self.state: GameState | None = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = make_initial_state()
        obs = np.array(encode_state(self.state), dtype=np.int64)
        mask = np.array(get_legal_mask(self.state, self.board, self.action_table), dtype=bool)
        return obs, {"action_mask": mask}

    def shaping_reward(self, old_state, new_state) -> float:
        active = old_state.active_player
        reward = 0.0
        for player in range(3):
            for i in range(4):
                old_pawn = old_state.pawns[player][i]
                new_pawn = new_state.pawns[player][i]
                if player != active and old_pawn.zone != Zone.BASE and new_pawn.zone == Zone.BASE:
                    reward += CAPTURE_REWARD
                if player == active and old_pawn.zone != Zone.HOME and new_pawn.zone == Zone.HOME:
                    reward += HOME_REWARD
        return reward

    def step(self, action_index: int):
        action = decode_action(self.action_table, action_index, self.state, self.board)
        new_state, game_over = advance_turn(self.state, action)

        reward = 1.0 if game_over else self.shaping_reward(self.state, new_state)

        self.state = new_state
        obs = np.array(encode_state(self.state), dtype=np.int64)

        if game_over:
            mask = np.zeros(len(self.action_table), dtype=bool)
        else:
            mask = np.array(get_legal_mask(self.state, self.board, self.action_table), dtype=bool)

        return obs, reward, game_over, False, {"action_mask": mask}

    def render(self):
        pass

if __name__ == "__main__":
    game_length_total = 0

    for i in range(100):
        env = TockEnv()
        obs, info = env.reset()

        done = False
        steps = 0
        while not done:
            mask = info["action_mask"]
            legal_indicies = [i for i, m in enumerate(mask) if m]
            action = random.choice(legal_indicies)

            obs, reward, done, _, info = env.step(action)
            steps += 1

        print(f"Game over in {steps} steps, reward={reward}")
        print(f"Obs shape: {obs.shape}, range: [{obs.min()}, {obs.max()}]")
        print(f"Mask length: {len(mask)}")

        game_length_total += steps

    print(f"Average game length: {game_length_total / 100}")