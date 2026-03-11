from action import Action, ActionType
from board import Board
from state import Zone, Pawn, GameState
from typing import Optional
import copy


def resolve_move(state: GameState, pawn: Pawn, path: list, is_seven=False) -> GameState:
    new_state = copy.deepcopy(state)
    moving_pawn = new_state.pawns[pawn.owner][pawn.pawn_id]
    tile_map = build_tile_map(new_state)
    if is_seven:
        capture_tiles = path.copy()
    else:
        capture_tiles = [path[-1]]

    for tile in capture_tiles:
        if tile_map[tile] is not None:
            tile_map[tile].zone = Zone.BASE
            tile_map[tile].index = None

    if path[-1] >= 54:
        moving_pawn.zone = Zone.HOME
    moving_pawn.index = path[-1]

    return new_state

def resolve_deploy(state: GameState) -> GameState:
    new_state = copy.deepcopy(state)
    # Assumes at least one BASE pawn exists — legal move generator should validate
    deploy_pawn = next(p for p in new_state.pawns[state.active_player] if p.zone == Zone.BASE)
    tile_map = build_tile_map(new_state)

    capture_tile_pawn = tile_map[Board.JUST_OUT_TILES[state.active_player]]
    if capture_tile_pawn is not None:
        capture_tile_pawn.zone = Zone.BASE
        capture_tile_pawn.index = None

    deploy_pawn.zone = Zone.MAIN
    deploy_pawn.index = Board.JUST_OUT_TILES[state.active_player]

    return new_state

def resolve_swap(state: GameState, pawn_one: Pawn, pawn_two: Pawn) -> GameState:
    new_state = copy.deepcopy(state)

    new_pawn_one = new_state.pawns[pawn_one.owner][pawn_one.pawn_id]
    new_pawn_two = new_state.pawns[pawn_two.owner][pawn_two.pawn_id]

    temp_idx = new_pawn_one.index
    new_pawn_one.index = new_pawn_two.index
    new_pawn_two.index = temp_idx

    return new_state

def build_tile_map(state: GameState) -> dict[int, Optional[Pawn]]:
    pawn_list = [pawn for player_pawns in state.pawns for pawn in player_pawns]
    tile_map = {i:None for i in range(66)}

    for pawn in pawn_list:
        if pawn.zone != Zone.BASE:
            tile_map[pawn.index] = pawn

    return tile_map

def is_path_blocked(path, tile_map, board):
    for tile in path:
        if tile_map[tile] is not None and board.is_just_out(tile_map[tile]):
            return True
    return False

def add_normal_move_actions(card, pawn, board, tile_map, state, actions, steps=None):
    steps = steps or card

    try:
        path = board.get_path(pawn, steps)
    except ValueError:
        return

    if not is_path_blocked(path, tile_map, board):
        actions.append(Action(card=card, action_type=ActionType.MOVE, pawn=pawn, path=path, enter_home=False))

    if board.HOME_ENTRY_TILES[pawn.owner] in path:
        try:
            home_path = board.get_path(pawn, steps, enter_home=True)
            if not is_path_blocked(home_path, tile_map, board):
                actions.append(Action(card=card, action_type=ActionType.MOVE, pawn=pawn, path=home_path, enter_home=True))
        except ValueError:
            pass

def get_legal_moves(state: GameState, board: Board) -> list[Action]:
    current_player = state.active_player
    player_hand = state.hands[current_player]

    actions = []
    tile_map = build_tile_map(state)

    player_pawns = state.pawns[current_player]
    other_player_index = [p for p in range(3) if p != current_player]
    other_pawns = [j for i in other_player_index for j in state.pawns[i]]
    base_pawns = [p for p in player_pawns if p.zone == Zone.BASE]

    for card in set(player_hand):
        if card == 1:
            if len(base_pawns) > 0:
                just_out_occupant = tile_map[board.JUST_OUT_TILES[current_player]]
                if just_out_occupant is None or just_out_occupant.owner != current_player:
                    actions.append(Action(card=card, action_type=ActionType.DEPLOY))

            for pawn in [p for p in player_pawns if p.zone != Zone.BASE]:
                add_normal_move_actions(1, pawn, board, tile_map, state, actions, steps=1)
                add_normal_move_actions(1, pawn, board, tile_map, state, actions, steps=11)
        elif card in [2, 3, 6, 8, 9, 10, 12]:
            for pawn in [p for p in player_pawns if p.zone != Zone.BASE]:
                add_normal_move_actions(card, pawn, board, tile_map, state, actions)
        elif card == 4:
            for pawn in [p for p in player_pawns if p.zone == Zone.MAIN]:
                path = board.get_path_backward(pawn, 4, )
                if not is_path_blocked(path, tile_map, board):
                    actions.append(
                        Action(card=card, action_type=ActionType.MOVE, pawn=pawn, path=path, enter_home=False))

        elif card == 5:
            for pawn in [p for j in state.pawns for p in j if p.zone != Zone.BASE and not board.is_just_out(p)]:
                add_normal_move_actions(card, pawn, board, tile_map, state, actions)
        elif card == 7:
            pass
        elif card == 11:
            for pawn in [p for p in player_pawns if p.zone == Zone.MAIN and not board.is_just_out(p)]:
                for other_pawn in [p for p in other_pawns if p.zone == Zone.MAIN and not board.is_just_out(p)]:
                    actions.append(Action(card=card, action_type=ActionType.SWAP, pawn=pawn, target_pawn=other_pawn))
        elif card == 13:
            if len(base_pawns) > 0:
                just_out_occupant = tile_map[board.JUST_OUT_TILES[current_player]]
                if just_out_occupant is None or just_out_occupant.owner != current_player:
                    actions.append(Action(card=card, action_type=ActionType.DEPLOY))

            for pawn in [p for p in player_pawns if p.zone != Zone.BASE]:
                add_normal_move_actions(card, pawn, board, tile_map, state, actions)

        else:
            raise Exception(f'Unknown card {card}')

