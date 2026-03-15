import unittest

from encoding import encode_state, decode_action, get_legal_mask, ACTION_TABLE
from rules import get_legal_moves
from action import ActionType
from board import Board
from state import Pawn, Zone, GameState


def make_pawn(owner, pawn_id, index, zone=Zone.MAIN):
    return Pawn(owner=owner, pawn_id=pawn_id, zone=zone, index=index)


def all_base_pawns():
    return [[make_pawn(p, i, None, Zone.BASE) for i in range(4)] for p in range(3)]


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
        deal_starting_player=deal_starting_player,
    )


def action_matches(a, b):
    """Semantically compare two Action objects."""
    if a.action_type != b.action_type or a.card != b.card:
        return False
    if a.action_type in (ActionType.DEPLOY, ActionType.DISCARD):
        return True
    if a.action_type == ActionType.MOVE:
        return (
            a.pawn.owner == b.pawn.owner
            and a.pawn.pawn_id == b.pawn.pawn_id
            and a.path == b.path
        )
    if a.action_type == ActionType.SWAP:
        pair_a = frozenset([(a.pawn.owner, a.pawn.pawn_id), (a.target_pawn.owner, a.target_pawn.pawn_id)])
        pair_b = frozenset([(b.pawn.owner, b.pawn.pawn_id), (b.target_pawn.owner, b.target_pawn.pawn_id)])
        return pair_a == pair_b
    if a.action_type == ActionType.SEVEN:
        moves_a = frozenset((p.owner, p.pawn_id, tuple(path)) for p, path in a.seven_moves)
        moves_b = frozenset((p.owner, p.pawn_id, tuple(path)) for p, path in b.seven_moves)
        return moves_a == moves_b
    return False


def matches_some_legal_action(decoded, legal_actions):
    return any(action_matches(decoded, la) for la in legal_actions)


def verify_round_trip(state, board, test_case):
    """Assert every mask-True action decodes to a legal action. Returns (true_indices, legal_actions)."""
    legal_actions = get_legal_moves(state, board)
    mask = get_legal_mask(state, board, ACTION_TABLE)
    true_indices = [i for i, m in enumerate(mask) if m]

    for idx in true_indices:
        decoded = decode_action(ACTION_TABLE, idx, state, board)
        test_case.assertTrue(
            matches_some_legal_action(decoded, legal_actions),
            f"Decoded action at index {idx} ({decoded}) doesn't match any legal action",
        )

    return true_indices, legal_actions


class TestEncodingRoundTrip(unittest.TestCase):

    def setUp(self):
        self.board = Board()

    # ── Basic movement ────────────────────────────────────────────────

    def test_basic_movement_round_trip(self):
        """Single pawn, one forward card → exactly one mask bit, decodes correctly."""
        pawns = all_base_pawns()
        pawns[0][0] = make_pawn(0, 0, 5)
        state = make_state(pawns, [3])
        true_indices, _ = verify_round_trip(state, self.board, self)
        self.assertEqual(len(true_indices), 1)

    def test_mask_length_equals_action_table(self):
        """Mask is always as long as ACTION_TABLE."""
        pawns = all_base_pawns()
        pawns[0][0] = make_pawn(0, 0, 5)
        state = make_state(pawns, [3])
        mask = get_legal_mask(state, self.board, ACTION_TABLE)
        self.assertEqual(len(mask), len(ACTION_TABLE))

    # ── Seven card ────────────────────────────────────────────────────

    def test_seven_single_pawn_round_trip(self):
        """Seven with one pawn on main track — straight move round-trips."""
        pawns = all_base_pawns()
        pawns[0][0] = make_pawn(0, 0, 5)
        state = make_state(pawns, [7])
        verify_round_trip(state, self.board, self)

    def test_seven_split_two_pawns_round_trip(self):
        """Seven split across two pawns round-trips correctly."""
        pawns = all_base_pawns()
        pawns[0][0] = make_pawn(0, 0, 5)
        pawns[0][1] = make_pawn(0, 1, 20)
        state = make_state(pawns, [7])
        verify_round_trip(state, self.board, self)

    # ── Card 5 ────────────────────────────────────────────────────────

    def test_card5_opponent_pawn_two_actions(self):
        """Opponent pawn crossing home entry with card 5 → two legal actions (enter/bypass)."""
        pawns = all_base_pawns()
        # Player 1's home entry is at 34; pawn at 32, 5 steps → path crosses 34.
        pawns[1][0] = make_pawn(1, 0, 32)
        state = make_state(pawns, [5])
        legal_actions = get_legal_moves(state, self.board)
        opponent_moves = [a for a in legal_actions if a.pawn.owner == 1]
        self.assertEqual(len(opponent_moves), 2)
        enter_flags = {a.enter_home for a in opponent_moves}
        self.assertEqual(enter_flags, {True, False})
        verify_round_trip(state, self.board, self)

    def test_card5_own_pawn_one_action(self):
        """Own pawn crossing home entry with card 5 → one legal action (deterministic enter)."""
        pawns = all_base_pawns()
        # Player 0's home entry is 16; pawn at 14, 5 steps → enters home at 57 (no overshoot).
        pawns[0][0] = make_pawn(0, 0, 14)
        state = make_state(pawns, [5])
        legal_actions = get_legal_moves(state, self.board)
        own_moves = [a for a in legal_actions if a.pawn.owner == 0]
        self.assertEqual(len(own_moves), 1)
        self.assertTrue(own_moves[0].enter_home)
        verify_round_trip(state, self.board, self)

    # ── Jack swap ─────────────────────────────────────────────────────

    def test_jack_swap_round_trip(self):
        """Jack swap between own and opponent pawn (not just-out) round-trips."""
        pawns = all_base_pawns()
        pawns[0][0] = make_pawn(0, 0, 5)
        pawns[1][0] = make_pawn(1, 0, 20)
        state = make_state(pawns, [11])
        true_indices, _ = verify_round_trip(state, self.board, self)
        # One distinct swap (A↔B and B↔A collapse to one mask bit)
        self.assertEqual(len(true_indices), 1)

    def test_jack_own_to_own_swap_round_trip(self):
        """Jack own-to-own swap is legal and round-trips."""
        pawns = all_base_pawns()
        pawns[0][0] = make_pawn(0, 0, 5)
        pawns[0][1] = make_pawn(0, 1, 10)
        pawns[1][0] = make_pawn(1, 0, 20)
        state = make_state(pawns, [11])
        verify_round_trip(state, self.board, self)

    # ── Forced discard ────────────────────────────────────────────────

    def test_forced_discard_only_discard_entries(self):
        """No legal moves → mask has only the discard entry set."""
        pawns = all_base_pawns()
        state = make_state(pawns, [3])
        mask = get_legal_mask(state, self.board, ACTION_TABLE)
        true_indices = [i for i, m in enumerate(mask) if m]
        self.assertEqual(len(true_indices), 1)
        action_type, card, *_ = ACTION_TABLE[true_indices[0]]
        self.assertEqual(action_type, ActionType.DISCARD)
        self.assertEqual(card, 3)

    def test_forced_discard_multiple_cards(self):
        """Multiple cards in hand with no moves → one discard entry per card."""
        pawns = all_base_pawns()
        state = make_state(pawns, [[3, 6, 9], [2], [2]])
        mask = get_legal_mask(state, self.board, ACTION_TABLE)
        true_indices = [i for i, m in enumerate(mask) if m]
        self.assertEqual(len(true_indices), 3)
        for idx in true_indices:
            self.assertEqual(ACTION_TABLE[idx][0], ActionType.DISCARD)

    def test_skip_flag_discard_only(self):
        """skip_flag=True → only discard entries regardless of board state."""
        pawns = all_base_pawns()
        pawns[0][0] = make_pawn(0, 0, 5)
        state = make_state(pawns, [3], skip_flag=True)
        mask = get_legal_mask(state, self.board, ACTION_TABLE)
        true_indices = [i for i, m in enumerate(mask) if m]
        for idx in true_indices:
            self.assertEqual(ACTION_TABLE[idx][0], ActionType.DISCARD)

    # ── Home entry overshoot ──────────────────────────────────────────

    def test_home_entry_overshoot_bypass(self):
        """Pawn where entering home would overshoot → bypass path used, enter_home=False."""
        pawns = all_base_pawns()
        # Pawn at 15, card 12: crosses home entry 16.
        # Entering home: 15→16(entry)→54, then 10 more = 64 > 57 (HOME_STRETCH_END). Overshoot.
        # So bypass path [16..27] with enter_home=False should be the only action.
        pawns[0][0] = make_pawn(0, 0, 15)
        state = make_state(pawns, [12])
        legal_actions = get_legal_moves(state, self.board)
        move_actions = [a for a in legal_actions if a.action_type == ActionType.MOVE]
        self.assertEqual(len(move_actions), 1)
        self.assertFalse(move_actions[0].enter_home)
        verify_round_trip(state, self.board, self)

    def test_home_entry_no_overshoot_enters(self):
        """Pawn where entering home fits exactly → deterministic enter, enter_home=True."""
        pawns = all_base_pawns()
        # Pawn at 13, card 3: path [14,15,16]. Crosses entry 16 at last step.
        # Enter home: 13→14→15→16(entry)→54. 3 steps, lands at 54. Fine.
        pawns[0][0] = make_pawn(0, 0, 13)
        state = make_state(pawns, [3])
        legal_actions = get_legal_moves(state, self.board)
        move_actions = [a for a in legal_actions if a.action_type == ActionType.MOVE]
        self.assertEqual(len(move_actions), 1)
        self.assertTrue(move_actions[0].enter_home)
        verify_round_trip(state, self.board, self)

    # ── Perspective rotation ──────────────────────────────────────────

    def test_perspective_rotation_produces_different_observations(self):
        """Same physical board with different active players gives different encodings."""
        pawns = all_base_pawns()
        pawns[0][0] = make_pawn(0, 0, 5)
        pawns[1][0] = make_pawn(1, 0, 23)

        state0 = make_state(pawns, [[3], [3], [3]], active_player=0)
        state1 = make_state(pawns, [[3], [3], [3]], active_player=1)

        enc0 = encode_state(state0)
        enc1 = encode_state(state1)
        self.assertNotEqual(enc0, enc1)

    def test_perspective_rotation_own_pawn_always_first(self):
        """Active player's pawn always occupies the first pawn slot in the encoding."""
        pawns = all_base_pawns()
        # Player 0 pawn at 5, player 1 pawn at 23 (relative from player 1 = (23-18)%54 = 5).
        pawns[0][0] = make_pawn(0, 0, 5)
        pawns[1][0] = make_pawn(1, 0, 23)

        state0 = make_state(pawns, [[3], [3], [3]], active_player=0)
        state1 = make_state(pawns, [[3], [3], [3]], active_player=1)

        enc0 = encode_state(state0)
        enc1 = encode_state(state1)

        # From player 0's view: pawn 0 at (5 - 0*18) % 54 = 5
        self.assertEqual(enc0[0], 5)
        # From player 1's view: pawn 0 at (23 - 1*18) % 54 = 5
        self.assertEqual(enc1[0], 5)

    def test_encoding_length(self):
        """encode_state always returns a 30-element vector."""
        pawns = all_base_pawns()
        pawns[0][0] = make_pawn(0, 0, 5)
        state = make_state(pawns, [[3, 2], [2], [2]])
        self.assertEqual(len(encode_state(state)), 30)


if __name__ == "__main__":
    unittest.main()