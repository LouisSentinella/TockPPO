import random

import gymnasium as gym
import numpy as np
import torch

from board import Board
from encoding import encode_state, decode_action, get_legal_mask, ACTION_TABLE
from rules import advance_turn, get_legal_moves
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


CAPTURE_REWARD = 0.02
HOME_REWARD    = 0.02


class TockEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, opponent_weights=None, opponent_weights2=None):
        super().__init__()
        self.board        = Board()
        self.action_table = ACTION_TABLE
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(89,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(len(ACTION_TABLE))
        self.state: GameState | None = None

        self._opponents = [None, None]  # index 0 → player 1, index 1 → player 2
        for slot, weights in enumerate([opponent_weights, opponent_weights2]):
            if weights is not None:
                from model import Agent
                opp = Agent()
                opp.load_state_dict(weights)
                opp.eval()
                self._opponents[slot] = opp

    def _opponent_action(self, state: GameState):
        slot = state.active_player - 1   # player 1 → slot 0, player 2 → slot 1
        opp  = self._opponents[slot]
        if opp is None:
            return random.choice(get_legal_moves(state, self.board))
        obs  = torch.as_tensor(encode_state(state), dtype=torch.float32).unsqueeze(0)
        mask = torch.tensor(
            get_legal_mask(state, self.board, self.action_table), dtype=torch.bool
        ).unsqueeze(0)
        with torch.no_grad():
            action_idx, _, _, _ = opp.get_action_and_value(obs, mask)
        return decode_action(self.action_table, action_idx.item(), state, self.board)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = make_initial_state()
        obs  = np.array(encode_state(self.state), dtype=np.float32)
        mask = np.array(get_legal_mask(self.state, self.board, self.action_table), dtype=bool)
        return obs, {"action_mask": mask}

    def shaping_reward(self, old_state: GameState, new_state: GameState) -> float:
        reward = 0.0
        for player in range(3):
            for i in range(4):
                old_pawn = old_state.pawns[player][i]
                new_pawn = new_state.pawns[player][i]
                # Player 0 captured an opponent pawn
                if player != 0 and old_pawn.zone != Zone.BASE and new_pawn.zone == Zone.BASE:
                    reward += CAPTURE_REWARD
                # Player 0 got a pawn home
                if player == 0 and old_pawn.zone != Zone.HOME and new_pawn.zone == Zone.HOME:
                    reward += HOME_REWARD
        return reward

    def step(self, action_index: int):
        action    = decode_action(self.action_table, action_index, self.state, self.board)
        new_state, game_over = advance_turn(self.state, action)

        reward = self.shaping_reward(self.state, new_state)

        if game_over:
            reward += 1.0
            self.state = new_state
            obs  = np.array(encode_state(self.state), dtype=np.float32)
            mask = np.zeros(len(self.action_table), dtype=bool)
            return obs, reward, True, False, {"action_mask": mask}

        self.state = new_state

        while self.state.active_player != 0:
            opp_action = self._opponent_action(self.state)
            self.state, game_over = advance_turn(self.state, opp_action)

            if game_over:
                # An opponent won — player 0 loses.
                reward -= 1.0
                obs  = np.array(encode_state(self.state), dtype=np.float32)
                mask = np.zeros(len(self.action_table), dtype=bool)
                return obs, reward, True, False, {"action_mask": mask}

        obs  = np.array(encode_state(self.state), dtype=np.float32)
        mask = np.array(get_legal_mask(self.state, self.board, self.action_table), dtype=bool)
        return obs, reward, False, False, {"action_mask": mask}

    def render(self):
        pass


if __name__ == "__main__":
    game_length_total = 0

    for i in range(100):
        env = TockEnv()
        obs, info = env.reset()

        done  = False
        steps = 0
        while not done:
            mask         = info["action_mask"]
            legal_indices = [i for i, m in enumerate(mask) if m]
            action       = random.choice(legal_indices)

            obs, reward, done, _, info = env.step(action)
            steps += 1

        print(f"Game over in {steps} steps, reward={reward}")
        game_length_total += steps

    print(f"Average game length: {game_length_total / 100}")