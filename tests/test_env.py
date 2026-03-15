import unittest

import encoding
import rules
from action import ActionType, Action
from board import Board
from state import Pawn, Zone, GameState
import copy

def make_pawn(owner, pawn_id, index, zone=Zone.MAIN):
    return Pawn(owner=owner, pawn_id=pawn_id, zone=zone, index=index)

def make_state(pawns, hands, active_player=0, skip_flag=False, deal_round=1, deal_starting_player=0):
    if hands and not isinstance(hands[0], list):
        hands = [hands if i == active_player else [2] for i in range(3)]
    return GameState(
        pawns=pawns,
        hands=hands,
        deck=list(range(1, 14)) * 4,
        discard_pile=[],
        active_player=active_player,
        skip_flag=skip_flag,
        deal_round=deal_round,
        deal_starting_player=deal_starting_player
    )

def all_base_pawns():
    return [[make_pawn(p, i, None, Zone.BASE) for i in range(4)] for p in range(3)]

class TestEnv(unittest.TestCase):

    def setUp(self):
        self.board = Board()

    # ── Basic turn advancement ────────────────────────────────────────

    def test_encoding_length(self):
        pawns = all_base_pawns()
        pawns[0][0] = make_pawn(0, 0, 5)
        state = make_state(pawns, [[3, 2], [2], [2]])
        self.assertEqual(len(encoding.encode_state(state)), 30)

if __name__ == '__main__':
    unittest.main()