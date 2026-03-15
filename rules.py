import random

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
        capture_tiles = [tile for tile in path.copy()[:-1] if tile_map[tile] is not None and tile_map[tile].owner != pawn.owner] + [path.copy()[-1]]
    else:
        capture_tiles = [path[-1]]

    for tile in capture_tiles:
        if tile_map[tile] is not None:
            tile_map[tile].zone = Zone.BASE
            tile_map[tile].index = None

    if path[-1] >= 54:
        moving_pawn.zone = Zone.HOME
    moving_pawn.index = path[-1]
    moving_pawn.is_newly_deployed = False

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
    deploy_pawn.is_newly_deployed = True

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

        occupant = tile_map[tile]
        if occupant is not None:
            if board.is_just_out(occupant):
                return True
            if occupant.zone == Zone.HOME:
                return True
    return False

def add_normal_move_actions(card, pawn, board, tile_map, state, actions, steps=None):
    steps = steps or card

    try:
        path = board.get_path(pawn, steps)
    except ValueError:
        return

    crosses_home = board.HOME_ENTRY_TILES[pawn.owner] in path

    if crosses_home:
        if pawn.owner == state.active_player:
            try:
                home_path = board.get_path(pawn, steps, enter_home=True)
                if not is_path_blocked(home_path, tile_map, board):
                    actions.append(Action(card=card, action_type=ActionType.MOVE, pawn=pawn, path=home_path, enter_home=True))
                    return
            except ValueError:
                pass
            if not is_path_blocked(path, tile_map, board):
                actions.append(Action(card=card, action_type=ActionType.MOVE, pawn=pawn, path=path, enter_home=False))
        else:
            if not is_path_blocked(path, tile_map, board):
                actions.append(Action(card=card, action_type=ActionType.MOVE, pawn=pawn, path=path, enter_home=False))
            try:
                home_path = board.get_path(pawn, steps, enter_home=True)
                if not is_path_blocked(home_path, tile_map, board):
                    actions.append(Action(card=card, action_type=ActionType.MOVE, pawn=pawn, path=home_path, enter_home=True))
            except ValueError:
                pass
    else:
        if not is_path_blocked(path, tile_map, board):
            actions.append(Action(card=card, action_type=ActionType.MOVE, pawn=pawn, path=path, enter_home=False))


def generate_seven_moves(remaining, moves_so_far, current_state, board, actions):

    if remaining == 0:
        actions.append(Action(card=7, action_type=ActionType.SEVEN, seven_moves=moves_so_far))
        return

    current_player = current_state.active_player
    tile_map = build_tile_map(current_state)

    used_pawn_ids = {move[0].pawn_id for move in moves_so_far}

    for pawn in [p for p in current_state.pawns[current_player] if p.zone != Zone.BASE]:

        if pawn.pawn_id in used_pawn_ids:
            continue

        for steps in range(1, remaining + 1):

            path = None

            # branch 1: no home entry
            try:
                candidate_path = board.get_path(pawn, steps)
            except ValueError:
                continue

            # branch 2: enter home if path crosses home entry
            if board.HOME_ENTRY_TILES[pawn.owner] in candidate_path:
                try:
                    home_path = board.get_path(pawn, steps, enter_home=True)
                    if not is_path_blocked(home_path, tile_map, board):
                        path = home_path
                except ValueError:
                    pass  # overshoot — fall through to use bypass path

            if path is None:
                if not is_path_blocked(candidate_path, tile_map, board):
                    path = candidate_path

            if path is not None:
                new_state = resolve_move(current_state, pawn, path, is_seven=True)
                generate_seven_moves(remaining - steps, moves_so_far + [(pawn, path)], new_state, board, actions)



def get_legal_moves(state: GameState, board: Board) -> list[Action]:
    current_player = state.active_player
    player_hand = state.hands[current_player]

    actions = []
    tile_map = build_tile_map(state)

    player_pawns = state.pawns[current_player]
    other_player_index = [p for p in range(3) if p != current_player]
    other_pawns = [j for i in other_player_index for j in state.pawns[i]]
    base_pawns = [p for p in player_pawns if p.zone == Zone.BASE]

    if state.skip_flag:
        return [Action(card=card, action_type=ActionType.DISCARD) for card in player_hand]

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
            generate_seven_moves(7, [], state, board, actions)
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

    if len(actions) == 0:
        for card in player_hand:
            actions.append(Action(card=card, action_type=ActionType.DISCARD))

    return actions

def advance_turn(state: GameState, action: Action) -> tuple[GameState, bool]:

    # Apply the action
    if action.action_type == ActionType.DISCARD:
        new_state = copy.deepcopy(state)
    elif action.action_type == ActionType.MOVE:
        new_state = resolve_move(state, action.pawn, action.path)
    elif action.action_type == ActionType.SWAP:
        new_state = resolve_swap(state, action.pawn, action.target_pawn)
    elif action.action_type == ActionType.DEPLOY:
        new_state = resolve_deploy(state)
    elif action.action_type == ActionType.SEVEN:
        new_state = state
        for pawn, path in action.seven_moves:
            new_state = resolve_move(new_state, pawn, path, is_seven=True)
    else:
        raise Exception(f'Unknown action type {action.action_type}')

    new_state.skip_flag = False

    # Remove played card, add to discard
    new_state.hands[new_state.active_player].remove(action.card)
    new_state.discard_pile.append(action.card)

    # Check win condition on player
    win_condition = True
    for pawn in new_state.pawns[new_state.active_player]:
        if pawn.zone != Zone.HOME:
            win_condition = False

    if win_condition:
        return new_state, True

    # Handle 10 card
    if action.card == 10:
        new_state.skip_flag = True

    # Advance active player
    new_state.active_player = (new_state.active_player + 1) % 3

    # Handle empty hands
    all_empty = True
    for hand in new_state.hands:
        if len(hand) > 0:
            all_empty = False

    if all_empty:
        new_state.deal_round += 1
        if new_state.deal_round > 4:
            new_state.deal_round = 1
            new_state.deal_starting_player = (new_state.deal_starting_player + 1) % 3
            new_state.deck = new_state.discard_pile.copy() + new_state.deck
            new_state.discard_pile = []
            random.shuffle(new_state.deck)

        if new_state.deal_round == 1:
            deal_size = 5
        else:
            deal_size = 4

        for i in range(3):
            player = (new_state.deal_starting_player + i) % 3
            new_state.hands[player] = new_state.deck[:deal_size]
            new_state.deck = new_state.deck[deal_size:]

    # Return new state
    return new_state, win_condition
