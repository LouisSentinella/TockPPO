from action import Action
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

def get_legal_moves(state: GameState, board: Board) -> list[Action]:
    current_player = state.active_player
    player_hand = state.hands[current_player]

    actions = []
    tile_map = build_tile_map(state)

    for card in set(player_hand):
        if card == 1:
            pass
        elif card in [2, 3, 6, 8, 9, 11, 12]:
            pass
        elif card == 4:
            pass
        elif card == 5:
            pass
        elif card == 7:
            pass
        elif card == 10:
            pass
        elif card == 13:
            pass
        else:
            raise Exception(f'Unknown card {card}')

