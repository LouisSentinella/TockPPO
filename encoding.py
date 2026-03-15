import math
import rules
from board import Board
from action import Action, ActionType
from rules import is_path_blocked
from state import GameState, Zone

# Observation layout (89 features total):
#   [0:72]  pawn positions — 12 pawns × 6 features each
#           per pawn: [is_base, sin(track), cos(track), track_norm, is_home, home_progress]
#   [72:85] card counts for active player's hand (ranks A–K, 13 values)
#   [85:87] opponent hand sizes (P1, P2 clockwise)
#   [87]    deal round
#   [88]    skip flag

def encode_state(state: GameState) -> list[float]:
    active_player = state.active_player
    clockwise_players = [active_player, (active_player + 1) % 3, (active_player + 2) % 3]

    encoding: list[float] = []
    # pawn positions — 6 features per pawn, 72 features total
    for player in clockwise_players:
        for pawn in state.pawns[player]:
            if pawn.zone == Zone.BASE:
                encoding.extend([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            elif pawn.zone == Zone.HOME:
                offset = pawn.index - Board.HOME_STRETCH_STARTING_INDEX[pawn.owner]
                encoding.extend([0.0, 0.0, 0.0, 0.0, 1.0, offset / 3.0])
            else:
                track_pos = (pawn.index - active_player * 18) % 54
                angle = 2.0 * math.pi * track_pos / 54.0
                encoding.extend([0.0, math.sin(angle), math.cos(angle), track_pos / 53.0, 0.0, 0.0])

    # card counts
    counts = [0] * 13
    for card in state.hands[active_player]:
        counts[card - 1] += 1
    encoding.extend(float(c) for c in counts)

    # opponent hand sizes
    for player in clockwise_players[1:]:
        encoding.append(float(len(state.hands[player])))

    # deal round
    encoding.append(float(state.deal_round))

    # skip flag
    encoding.append(float(state.skip_flag))

    return encoding


def generate_seven_splits() -> list[tuple]:
    results = []

    def recurse(remaining, current, used_pawn_ids):
        if remaining == 0:
            results.append(tuple(current))
            return
        for pawn_id in range(4):
            if pawn_id in used_pawn_ids:
                continue
            for steps in range(1, remaining + 1):
                recurse(remaining - steps, current + [(pawn_id, steps)], used_pawn_ids | {pawn_id})

    recurse(7, [], set())
    return results

SEVEN_SPLITS = generate_seven_splits()
SEVEN_SPLIT_REVERSE: dict[tuple, int] = {
    split: idx for idx, split in enumerate(SEVEN_SPLITS)
}

def build_action_table() -> list[tuple]:
    # action format -> (action_type, card, pawn_id, steps, enter_home, target_pawn_id, seven_split_id)
    action_table = []
    # Ace
    action_table.append((ActionType.DEPLOY, 1,  None, None, None, None, None))
    for i in range(4):
        action_table.append((ActionType.MOVE, 1, i, 1, None,None, None))
        action_table.append((ActionType.MOVE, 1, i, 11, None, None, None))

    # Cards
    for move_num in ( 2, 3, 6, 8, 9, 10, 12):
        for i in range(4):
            action_table.append((ActionType.MOVE, move_num, i, move_num, None, None, None))

    for i in range(4):
        action_table.append((ActionType.MOVE, 5, i, 5, None, None, None))
    for i in range(4, 12):
        action_table.append((ActionType.MOVE, 5, i, 5, False, None, None))
        action_table.append((ActionType.MOVE, 5, i, 5, True, None, None))

    for i in range(4):
        action_table.append((ActionType.MOVE, 4, i, 4, None,None, None))
    # K
    action_table.append((ActionType.DEPLOY, 13, None, None, None, None, None))
    for i in range(4):
        action_table.append((ActionType.MOVE, 13, i, 13, None, None, None))

    # J
    for i in range(4):
        for j in range(0, 12):
            if j > i:
                action_table.append((ActionType.SWAP, 11, i, None, None, j, None))

    # 7
    for split_id, split in enumerate(SEVEN_SPLITS):
        action_table.append((ActionType.SEVEN, 7, None, None, None, None, split_id))

    # Discard
    for card_value in range(1, 14):
        action_table.append((ActionType.DISCARD, card_value, None, None, None, None, None))

    return action_table

ACTION_TABLE: list[tuple] = build_action_table()

def build_action_lookup(action_table: list[tuple]) -> dict[tuple, int]:
    lookup: dict[tuple, int] = {}
    for idx, entry in enumerate(action_table):
        action_type, card, pawn_id, steps, enter_home, target_pawn_id, seven_split_id = entry

        if action_type == ActionType.DEPLOY:
            key = (ActionType.DEPLOY, card)
        elif action_type == ActionType.MOVE:
            key = (ActionType.MOVE, card, pawn_id, steps, enter_home)
        elif action_type == ActionType.SWAP:
            key = (ActionType.SWAP, card, pawn_id, target_pawn_id)
        elif action_type == ActionType.SEVEN:
            key = (ActionType.SEVEN, seven_split_id)
        elif action_type == ActionType.DISCARD:
            key = (ActionType.DISCARD, card)
        else:
            raise ValueError(f"Unknown action type in table: {action_type}")

        lookup[key] = idx

    return lookup


ACTION_LOOKUP: dict[tuple, int] = build_action_lookup(ACTION_TABLE)

def action_to_key(action: Action, state: GameState) -> tuple:
    if action.action_type == ActionType.DEPLOY:
        return (ActionType.DEPLOY, action.card)

    elif action.action_type == ActionType.DISCARD:
        return (ActionType.DISCARD, action.card)

    elif action.action_type == ActionType.SWAP:
        rel_pawn        = relativize_pawn(action.pawn,        state)
        rel_target_pawn = relativize_pawn(action.target_pawn, state)
        lo, hi = sorted((rel_pawn, rel_target_pawn))
        return (ActionType.SWAP, action.card, lo, hi)

    elif action.action_type == ActionType.MOVE:
        rel_pawn = relativize_pawn(action.pawn, state)
        steps    = len(action.path)

        if action.pawn.owner != state.active_player and action.card == 5:
            enter_home = action.enter_home
        else:
            enter_home = None
        return (ActionType.MOVE, action.card, rel_pawn, steps, enter_home)

    elif action.action_type == ActionType.SEVEN:
        split_as_pairs = tuple(
            (relativize_pawn(pawn, state), len(path))
            for pawn, path in action.seven_moves
        )
        split_id = SEVEN_SPLIT_REVERSE[split_as_pairs]
        return (ActionType.SEVEN, split_id)

    else:
        raise ValueError(f"Unknown action type: {action.action_type}")


def get_legal_mask(state: GameState, board: Board, action_table: list[tuple]) -> list[bool]:
    mask = [False] * len(action_table)

    legal_actions = rules.get_legal_moves(state, board)

    for action in legal_actions:
        key = action_to_key(action, state)
        idx = ACTION_LOOKUP.get(key)
        if idx is None:
            raise KeyError(
                f"Legal action produced a key not found in ACTION_LOOKUP: {key}\n"
                f"Action: {action}"
            )
        mask[idx] = True

    return mask


def relativize_pawn(pawn, state: GameState) -> int:
    active = state.active_player
    if pawn.owner == active:
        return pawn.pawn_id
    elif pawn.owner == (active + 1) % 3:
        return 4 + pawn.pawn_id
    else:
        return 8 + pawn.pawn_id

def resolve_pawn(pawn_id, state):
    if 0 <= pawn_id < 4:
        return state.pawns[state.active_player][pawn_id]
    elif 4 <= pawn_id < 8:
        return state.pawns[(state.active_player+1) % 3][pawn_id - 4]
    else:
        return state.pawns[(state.active_player+2) % 3][pawn_id - 8]

def decode_action(action_table, action_index, state, board) -> Action:
    action_type, card, pawn_id, steps, enter_home, target_pawn_id, seven_split_id = action_table[action_index]

    pawn = resolve_pawn(pawn_id, state) if pawn_id is not None else None

    if action_type == ActionType.DEPLOY:
        return Action(card=card, action_type=ActionType.DEPLOY)

    elif action_type == ActionType.DISCARD:
        return Action(card=card, action_type=ActionType.DISCARD)

    elif action_type == ActionType.SWAP:
        target = resolve_pawn(target_pawn_id, state)
        return Action(card=card, action_type=ActionType.SWAP, pawn=pawn, target_pawn=target)

    elif action_type == ActionType.MOVE:
        if card == 4:
            path = board.get_path_backward(pawn, 4)
            return Action(card=card, action_type=ActionType.MOVE, pawn=pawn, path=path)

        if enter_home is not None:
            path = board.get_path(pawn, steps, enter_home=enter_home)
            return Action(card=card, action_type=ActionType.MOVE, pawn=pawn, path=path, enter_home=enter_home)
        else:
            path = board.get_path(pawn, steps)
            if board.HOME_ENTRY_TILES[pawn.owner] in path:
                try:
                    home_path = board.get_path(pawn, steps, enter_home=True)
                    return Action(card=card, action_type=ActionType.MOVE, pawn=pawn, path=home_path, enter_home=True)
                except ValueError:
                    pass
            return Action(card=card, action_type=ActionType.MOVE, pawn=pawn, path=path, enter_home=False)

    elif action_type == ActionType.SEVEN:
        split = SEVEN_SPLITS[seven_split_id]
        seven_moves = []
        current_state = state

        for sub_pawn_id, sub_steps in split:
            sub_pawn = resolve_pawn(sub_pawn_id, current_state)
            tile_map = rules.build_tile_map(current_state)

            candidate_path = board.get_path(sub_pawn, sub_steps)
            path = None

            if board.HOME_ENTRY_TILES[sub_pawn.owner] in candidate_path:
                try:
                    home_path = board.get_path(sub_pawn, sub_steps, enter_home=True)
                    if not is_path_blocked(home_path, tile_map, board):
                        path = home_path
                except ValueError:
                    pass

            if path is None:
                path = candidate_path

            seven_moves.append((sub_pawn, path))
            current_state = rules.resolve_move(current_state, sub_pawn, path, is_seven=True)

        return Action(card=7, action_type=ActionType.SEVEN, seven_moves=seven_moves)

    else:
        raise Exception(f"Unknown action type: {action_type}")



