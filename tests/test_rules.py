import unittest

import rules
from board import Board
from state import Pawn, Zone, GameState
import copy

class TestBoard(unittest.TestCase):

    def setUp(self):
        player_zero_pawns = [
            Pawn(owner=0, pawn_id=0, zone=Zone.MAIN, index=5),
            Pawn(owner=0, pawn_id=1, zone=Zone.MAIN, index=20),
            Pawn(owner=0, pawn_id=2, zone=Zone.HOME, index=54),
            Pawn(owner=0, pawn_id=3, zone=Zone.BASE, index=None)
        ]
        player_one_pawns = [
            Pawn(owner=1, pawn_id=0, zone=Zone.MAIN, index=6),
            Pawn(owner=1, pawn_id=1, zone=Zone.MAIN, index=22),
            Pawn(owner=1, pawn_id=2, zone=Zone.MAIN, index=30),
            Pawn(owner=1, pawn_id=3, zone=Zone.BASE, index=None)
        ]
        player_two_pawns = [
            Pawn(owner=2, pawn_id=0, zone=Zone.MAIN, index=18),
            Pawn(owner=2, pawn_id=1, zone=Zone.MAIN, index=40),
            Pawn(owner=2, pawn_id=2, zone=Zone.HOME, index=62),
            Pawn(owner=2, pawn_id=3, zone=Zone.BASE, index=None)
        ]
        self.state = GameState(
            pawns=[player_zero_pawns, player_one_pawns, player_two_pawns],
            hands=[[2, 3, 5], [7, 1], [12, 4, 6]],
            deck=list(range(1, 14)) * 4,
            discard_pile=[],
            active_player=0,
            skip_flag=False,
            deal_round=1,
            deal_starting_player=0
        )

    def test_regular_movement(self):
        final_state = rules.resolve_move(self.state, self.state.pawns[0][0], [6, 7, 8])
        self.assertEqual(final_state.pawns[0][0].index, 8)
        self.assertEqual(final_state.pawns[0][0].zone, Zone.MAIN)

    def test_capture(self):
        final_state = rules.resolve_move(self.state, self.state.pawns[0][0], [6])
        self.assertEqual(final_state.pawns[0][0].index, 6)
        self.assertEqual(final_state.pawns[1][0].zone, Zone.BASE)
        self.assertIsNone(final_state.pawns[1][0].index)

    def test_capture_seven(self):
        final_state = rules.resolve_move(self.state, self.state.pawns[0][0], [6, 7, 8], is_seven=True)
        self.assertEqual(final_state.pawns[0][0].index, 8)
        self.assertEqual(final_state.pawns[1][0].zone, Zone.BASE)
        self.assertIsNone(final_state.pawns[1][0].index)

    def test_enter_home(self):
        new_state = copy.deepcopy(self.state)
        new_state.active_player = 1
        final_state = rules.resolve_move(new_state, new_state.pawns[1][2], [31, 32, 33, 34, 58, 59])
        self.assertEqual(final_state.pawns[1][2].index, 59)
        self.assertEqual(final_state.pawns[1][2].zone, Zone.HOME)

    def test_deploy(self):
        final_state = rules.resolve_deploy(self.state)
        self.assertEqual(final_state.pawns[0][3].zone, Zone.MAIN)
        self.assertEqual(final_state.pawns[0][3].index, 0)

    def test_deploy_capture(self):
        new_state = copy.deepcopy(self.state)
        new_state.active_player = 1
        final_state = rules.resolve_deploy(new_state)
        self.assertEqual(final_state.pawns[1][3].zone, Zone.MAIN)
        self.assertEqual(final_state.pawns[1][3].index, 18)
        self.assertEqual(final_state.pawns[2][3].zone, Zone.BASE)
        self.assertIsNone(final_state.pawns[2][3].index)

    def test_deploy_swap(self):
        final_state = rules.resolve_swap(self.state, self.state.pawns[0][0], self.state.pawns[1][1])
        self.assertEqual(final_state.pawns[0][0].index, 22)
        self.assertEqual(final_state.pawns[1][1].index, 5)

if __name__ == '__main__':
    unittest.main()