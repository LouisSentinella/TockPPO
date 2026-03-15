import unittest

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

class TestLegalMoves(unittest.TestCase):

    def setUp(self):
        self.board = Board()

    # ── Normal movement ──────────────────────────────────────────────

    def test_normal_move_generates_action(self):
        pawns = all_base_pawns()
        pawns[0][0] = make_pawn(0, 0, 5)
        state = make_state(pawns, [3])
        actions = rules.get_legal_moves(state, self.board)
        move_actions = [a for a in actions if a.action_type == ActionType.MOVE]
        self.assertEqual(len(move_actions), 1)
        self.assertEqual(move_actions[0].path[-1], 8)

    def test_base_pawns_excluded_from_movement(self):
        pawns = all_base_pawns()  # all pawns in base
        state = make_state(pawns, [3])
        actions = rules.get_legal_moves(state, self.board)
        move_actions = [a for a in actions if a.action_type == ActionType.MOVE]
        self.assertEqual(len(move_actions), 0)

    def test_home_stretch_pawn_can_move(self):
        pawns = all_base_pawns()
        pawns[0][0] = make_pawn(0, 0, 54, Zone.HOME)
        state = make_state(pawns, [2])
        actions = rules.get_legal_moves(state, self.board)
        move_actions = [a for a in actions if a.action_type == ActionType.MOVE]
        self.assertEqual(len(move_actions), 1)
        self.assertEqual(move_actions[0].path[-1], 56)

    def test_overshoot_home_stretch_excluded(self):
        pawns = all_base_pawns()
        pawns[0][0] = make_pawn(0, 0, 56, Zone.HOME)
        state = make_state(pawns, [3])  # 3 steps would overshoot (56+3=59 > 57)
        actions = rules.get_legal_moves(state, self.board)
        move_actions = [a for a in actions if a.action_type == ActionType.MOVE]
        self.assertEqual(len(move_actions), 0)

    # ── Blocking ─────────────────────────────────────────────────────

    def test_own_just_out_blocks_path(self):
        pawns = all_base_pawns()
        pawns[0][0] = make_pawn(0, 0, 50)  # close to wrap-around
        pawns[0][1] = Pawn(owner=0, pawn_id=1, zone=Zone.MAIN, index=0, is_newly_deployed=True)
        state = make_state(pawns, [6])  # path: 51,52,53,0,1,2 — passes through tile 0
        actions = rules.get_legal_moves(state, self.board)
        move_actions = [a for a in actions if a.action_type == ActionType.MOVE and a.pawn.pawn_id == 0]
        self.assertEqual(len(move_actions), 0)

    def test_opponent_just_out_does_not_block(self):
        pawns = all_base_pawns()
        pawns[0][0] = make_pawn(0, 0, 15)
        # Opponent newly deployed on tile 18
        pawns[1][0] = Pawn(owner=1, pawn_id=0, zone=Zone.MAIN, index=18, is_newly_deployed=True)
        state = make_state(pawns, [3])  # path: 16, 17, 18 — should NOT be blocked
        actions = rules.get_legal_moves(state, self.board)
        move_actions = [a for a in actions if a.action_type == ActionType.MOVE]
        self.assertTrue(len(move_actions) > 0)

    def test_home_stretch_pawn_blocks_path(self):
        pawns = all_base_pawns()
        pawns[0][0] = make_pawn(0, 0, 54, Zone.HOME)
        pawns[0][1] = make_pawn(0, 1, 55, Zone.HOME)
        state = make_state(pawns, [2])  # pawn at 54, 2 steps would reach 56 but passes 55
        actions = rules.get_legal_moves(state, self.board)
        move_actions = [a for a in actions if a.action_type == ActionType.MOVE and a.pawn.pawn_id == 0]
        self.assertEqual(len(move_actions), 0)

    # ── Deploy ───────────────────────────────────────────────────────

    def test_ace_deploy_when_just_out_empty(self):
        pawns = all_base_pawns()
        state = make_state(pawns, [1])
        actions = rules.get_legal_moves(state, self.board)
        deploy_actions = [a for a in actions if a.action_type == ActionType.DEPLOY]
        self.assertEqual(len(deploy_actions), 1)

    def test_ace_deploy_blocked_by_own_pawn(self):
        pawns = all_base_pawns()
        pawns[0][0] = Pawn(owner=0, pawn_id=0, zone=Zone.MAIN, index=0, is_newly_deployed=True)
        state = make_state(pawns, [1])
        actions = rules.get_legal_moves(state, self.board)
        deploy_actions = [a for a in actions if a.action_type == ActionType.DEPLOY]
        self.assertEqual(len(deploy_actions), 0)

    def test_ace_deploy_allowed_when_opponent_on_just_out(self):
        pawns = all_base_pawns()
        pawns[1][0] = make_pawn(1, 0, 0)  # opponent on player 0's just-out, not newly deployed
        state = make_state(pawns, [1])
        actions = rules.get_legal_moves(state, self.board)
        deploy_actions = [a for a in actions if a.action_type == ActionType.DEPLOY]
        self.assertEqual(len(deploy_actions), 1)

    def test_no_deploy_when_no_base_pawns(self):
        pawns = all_base_pawns()
        pawns[0] = [make_pawn(0, i, i + 5) for i in range(4)]
        state = make_state(pawns, [1])
        actions = rules.get_legal_moves(state, self.board)
        deploy_actions = [a for a in actions if a.action_type == ActionType.DEPLOY]
        self.assertEqual(len(deploy_actions), 0)

    # ── Card 4 ───────────────────────────────────────────────────────

    def test_card_4_backward(self):
        pawns = all_base_pawns()
        pawns[0][0] = make_pawn(0, 0, 10)
        state = make_state(pawns, [4])
        actions = rules.get_legal_moves(state, self.board)
        move_actions = [a for a in actions if a.action_type == ActionType.MOVE]
        self.assertEqual(len(move_actions), 1)
        self.assertEqual(move_actions[0].path[-1], 6)

    def test_card_4_excludes_home_pawns(self):
        pawns = all_base_pawns()
        pawns[0][0] = make_pawn(0, 0, 55, Zone.HOME)
        state = make_state(pawns, [4])
        actions = rules.get_legal_moves(state, self.board)
        move_actions = [a for a in actions if a.action_type == ActionType.MOVE]
        self.assertEqual(len(move_actions), 0)

    # ── Card 5 ───────────────────────────────────────────────────────

    def test_card_5_moves_opponent_pawn(self):
        pawns = all_base_pawns()
        pawns[1][0] = make_pawn(1, 0, 10)
        state = make_state(pawns, [5])
        actions = rules.get_legal_moves(state, self.board)
        move_actions = [a for a in actions if a.action_type == ActionType.MOVE]
        target_pawns = [a.pawn.owner for a in move_actions]
        self.assertIn(1, target_pawns)

    def test_card_5_excludes_just_out_pawns(self):
        pawns = all_base_pawns()
        pawns[1][0] = Pawn(owner=1, pawn_id=0, zone=Zone.MAIN, index=18, is_newly_deployed=True)
        state = make_state(pawns, [5])
        actions = rules.get_legal_moves(state, self.board)
        move_actions = [a for a in actions if a.action_type == ActionType.MOVE]
        targets = [a.pawn.index for a in move_actions]
        self.assertNotIn(18, targets)

    # ── Jack ─────────────────────────────────────────────────────────

    def test_jack_generates_swap(self):
        pawns = all_base_pawns()
        pawns[0][0] = make_pawn(0, 0, 5)
        pawns[1][0] = make_pawn(1, 0, 20)
        state = make_state(pawns, [11])
        actions = rules.get_legal_moves(state, self.board)
        swap_actions = [a for a in actions if a.action_type == ActionType.SWAP]
        self.assertEqual(len(swap_actions), 1)

    def test_jack_excludes_just_out_own(self):
        pawns = all_base_pawns()
        pawns[0][0] = Pawn(owner=0, pawn_id=0, zone=Zone.MAIN, index=0, is_newly_deployed=True)
        pawns[1][0] = make_pawn(1, 0, 20)
        state = make_state(pawns, [11])
        actions = rules.get_legal_moves(state, self.board)
        swap_actions = [a for a in actions if a.action_type == ActionType.SWAP]
        self.assertEqual(len(swap_actions), 0)

    def test_jack_excludes_just_out_opponent(self):
        pawns = all_base_pawns()
        pawns[0][0] = make_pawn(0, 0, 5)
        pawns[1][0] = Pawn(owner=1, pawn_id=0, zone=Zone.MAIN, index=18, is_newly_deployed=True)
        state = make_state(pawns, [11])
        actions = rules.get_legal_moves(state, self.board)
        swap_actions = [a for a in actions if a.action_type == ActionType.SWAP]
        self.assertEqual(len(swap_actions), 0)

    # ── Forced discard ───────────────────────────────────────────────

    def test_forced_discard_when_no_moves(self):
        pawns = all_base_pawns()  # all base, no moves possible with card 3
        state = make_state(pawns, [3])
        actions = rules.get_legal_moves(state, self.board)
        discard_actions = [a for a in actions if a.action_type == ActionType.DISCARD]
        self.assertEqual(len(discard_actions), 1)
        self.assertEqual(discard_actions[0].card, 3)

    def test_forced_discard_covers_all_cards_in_hand(self):
        pawns = all_base_pawns()
        state = make_state(pawns, [3, 6, 9])
        actions = rules.get_legal_moves(state, self.board)
        discard_actions = [a for a in actions if a.action_type == ActionType.DISCARD]
        self.assertEqual(len(discard_actions), 3)

    # ── Seven ────────────────────────────────────────────────────────

    def test_seven_straight(self):
        pawns = all_base_pawns()
        pawns[0][0] = make_pawn(0, 0, 5)
        state = make_state(pawns, [7])
        actions = rules.get_legal_moves(state, self.board)
        seven_actions = [a for a in actions if a.action_type == ActionType.SEVEN]
        straight = [a for a in seven_actions if len(a.seven_moves) == 1]
        self.assertTrue(len(straight) > 0)
        self.assertEqual(straight[0].seven_moves[0][1][-1], 12)

    def test_seven_split_two_pawns(self):
        pawns = all_base_pawns()
        pawns[0][0] = make_pawn(0, 0, 5)
        pawns[0][1] = make_pawn(0, 1, 20)
        state = make_state(pawns, [7])
        actions = rules.get_legal_moves(state, self.board)
        seven_actions = [a for a in actions if a.action_type == ActionType.SEVEN]
        splits = [a for a in seven_actions if len(a.seven_moves) == 2]
        self.assertTrue(len(splits) > 0)

    def test_seven_captures_intermediate(self):
        pawns = all_base_pawns()
        pawns[0][0] = make_pawn(0, 0, 5)
        pawns[1][0] = make_pawn(1, 0, 8)  # sits at intermediate tile
        state = make_state(pawns, [7])
        actions = rules.get_legal_moves(state, self.board)
        # A straight 7 from 5 passes through 8 — should still be legal
        seven_actions = [a for a in actions if a.action_type == ActionType.SEVEN]
        straight = [a for a in seven_actions if len(a.seven_moves) == 1 and a.seven_moves[0][1][-1] == 12]
        self.assertTrue(len(straight) > 0)

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

class TestAdvanceTurn(unittest.TestCase):

    def setUp(self):
        self.board = Board()

    # ── Basic turn advancement ────────────────────────────────────────

    def test_active_player_advances(self):
        pawns = all_base_pawns()
        pawns[0][0] = make_pawn(0, 0, 5)
        state = make_state(pawns, [[3, 2], [2], [2]])
        action = Action(card=3, action_type=ActionType.MOVE, pawn=state.pawns[0][0], path=[6, 7, 8])
        new_state, _ = rules.advance_turn(state, action)
        self.assertEqual(new_state.active_player, 1)

    def test_card_removed_from_hand(self):
        pawns = all_base_pawns()
        pawns[0][0] = make_pawn(0, 0, 5)
        state = make_state(pawns, [[3, 2], [2], [2]])
        action = Action(card=3, action_type=ActionType.MOVE, pawn=state.pawns[0][0], path=[6, 7, 8])
        new_state, _ = rules.advance_turn(state, action)
        self.assertNotIn(3, new_state.hands[0])

    def test_card_added_to_discard(self):
        pawns = all_base_pawns()
        pawns[0][0] = make_pawn(0, 0, 5)
        state = make_state(pawns, [[3, 2], [2], [2]])
        action = Action(card=3, action_type=ActionType.MOVE, pawn=state.pawns[0][0], path=[6, 7, 8])
        new_state, _ = rules.advance_turn(state, action)
        self.assertIn(3, new_state.discard_pile)

    def test_player_wraps_from_2_to_0(self):
        pawns = all_base_pawns()
        pawns[2][0] = make_pawn(2, 0, 5)
        state = make_state(pawns, [[2], [2], [3, 2]], active_player=2)
        action = Action(card=3, action_type=ActionType.MOVE, pawn=state.pawns[2][0], path=[6, 7, 8])
        new_state, _ = rules.advance_turn(state, action)
        self.assertEqual(new_state.active_player, 0)

    # ── Win condition ─────────────────────────────────────────────────

    def test_win_condition_true(self):
        pawns = all_base_pawns()
        pawns[0] = [
            make_pawn(0, 0, 54, Zone.HOME),
            make_pawn(0, 1, 55, Zone.HOME),
            make_pawn(0, 2, 56, Zone.HOME),
            make_pawn(0, 3, 56, Zone.MAIN),  # last pawn about to go home
        ]
        state = make_state(pawns, [[2], [2], [2]])
        action = Action(card=2, action_type=ActionType.MOVE, pawn=state.pawns[0][3], path=[57, 58])
        # path ends at 58 which is >= 54 so zone goes HOME...
        # Actually let's use a simpler setup
        pawns[0][3] = make_pawn(0, 3, 55, Zone.HOME)
        state = make_state(pawns, [[2], [2], [2]])
        # All 4 already home — force win by playing any move and checking
        pawns[0] = [make_pawn(0, i, 54 + i, Zone.HOME) for i in range(4)]
        state = make_state(pawns, [[2], [2], [2]])
        action = Action(card=2, action_type=ActionType.DISCARD)
        new_state, game_over = rules.advance_turn(state, action)
        self.assertTrue(game_over)

    def test_win_condition_false(self):
        pawns = all_base_pawns()
        pawns[0][0] = make_pawn(0, 0, 5)
        state = make_state(pawns, [[3, 2], [2], [2]])
        action = Action(card=3, action_type=ActionType.MOVE, pawn=state.pawns[0][0], path=[6, 7, 8])
        _, game_over = rules.advance_turn(state, action)
        self.assertFalse(game_over)

    # ── 10 card ───────────────────────────────────────────────────────

    def test_10_sets_skip_flag(self):
        pawns = all_base_pawns()
        pawns[0][0] = make_pawn(0, 0, 5)
        state = make_state(pawns, [[10, 2], [2], [2]])
        action = Action(card=10, action_type=ActionType.MOVE, pawn=state.pawns[0][0], path=[6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        new_state, _ = rules.advance_turn(state, action)
        self.assertTrue(new_state.skip_flag)

    def test_skip_flag_cleared_after_discard(self):
        pawns = all_base_pawns()
        state = make_state(pawns, [[2], [3, 2], [2]], active_player=1, skip_flag=True)
        action = Action(card=3, action_type=ActionType.DISCARD)
        new_state, _ = rules.advance_turn(state, action)
        self.assertFalse(new_state.skip_flag)

    # ── Redeal ───────────────────────────────────────────────────────

    def test_hands_redealt_when_empty(self):
        pawns = all_base_pawns()
        pawns[2][0] = make_pawn(2, 0, 5)
        state = make_state(pawns, [[], [], [3]], active_player=2, deal_round=1)
        action = Action(card=3, action_type=ActionType.MOVE, pawn=state.pawns[2][0], path=[6, 7, 8])
        new_state, _ = rules.advance_turn(state, action)
        self.assertTrue(all(len(h) > 0 for h in new_state.hands))

    def test_deal_round_increments(self):
        pawns = all_base_pawns()
        pawns[2][0] = make_pawn(2, 0, 5)
        state = make_state(pawns, [[], [], [3]], active_player=2, deal_round=1)
        action = Action(card=3, action_type=ActionType.MOVE, pawn=state.pawns[2][0], path=[6, 7, 8])
        new_state, _ = rules.advance_turn(state, action)
        self.assertEqual(new_state.deal_round, 2)

    def test_deal_round_resets_after_4(self):
        pawns = all_base_pawns()
        pawns[2][0] = make_pawn(2, 0, 5)
        state = make_state(pawns, [[], [], [3]], active_player=2, deal_round=4)
        state.discard_pile = list(range(1, 14)) * 4
        state.deck = []
        action = Action(card=3, action_type=ActionType.MOVE, pawn=state.pawns[2][0], path=[6, 7, 8])
        new_state, _ = rules.advance_turn(state, action)
        self.assertEqual(new_state.deal_round, 1)

    def test_deal_size_round_1(self):
        pawns = all_base_pawns()
        pawns[2][0] = make_pawn(2, 0, 5)
        state = make_state(pawns, [[], [], [3]], active_player=2, deal_round=4)
        state.discard_pile = list(range(1, 14)) * 4
        state.deck = []
        action = Action(card=3, action_type=ActionType.MOVE, pawn=state.pawns[2][0], path=[6, 7, 8])
        new_state, _ = rules.advance_turn(state, action)
        self.assertTrue(all(len(h) == 5 for h in new_state.hands))

    def test_deal_size_round_2(self):
        pawns = all_base_pawns()
        pawns[2][0] = make_pawn(2, 0, 5)
        state = make_state(pawns, [[], [], [3]], active_player=2, deal_round=1)
        action = Action(card=3, action_type=ActionType.MOVE, pawn=state.pawns[2][0], path=[6, 7, 8])
        new_state, _ = rules.advance_turn(state, action)
        self.assertTrue(all(len(h) == 4 for h in new_state.hands))

    # ── SEVEN ────────────────────────────────────────────────────────

    def test_seven_applies_all_submoves(self):
        pawns = all_base_pawns()
        pawns[0][0] = make_pawn(0, 0, 5)
        pawns[0][1] = make_pawn(0, 1, 20)
        state = make_state(pawns, [[7, 2], [2], [2]])
        seven_moves = [
            (state.pawns[0][0], [6, 7, 8]),
            (state.pawns[0][1], [21, 22, 23, 24])
        ]
        action = Action(card=7, action_type=ActionType.SEVEN, seven_moves=seven_moves)
        new_state, _ = rules.advance_turn(state, action)
        self.assertEqual(new_state.pawns[0][0].index, 8)
        self.assertEqual(new_state.pawns[0][1].index, 24)

if __name__ == '__main__':
    unittest.main()