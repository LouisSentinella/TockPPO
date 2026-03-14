from state import GameState, Zone

def encode_state(state: GameState) -> list[int]:
    active_player = state.active_player
    clockwise_players = [active_player, (active_player + 1) % 3, (active_player + 2) % 3]

    encoding = []
    # pawn positions
    for player in clockwise_players:
        for pawn in state.pawns[player]:
            if pawn.zone == Zone.BASE:
                encoding.append(-1)
            else:
                encoding.append(pawn.index)

    # card counts
    counts = [0] * 13
    for card in state.hands[active_player]:
        counts[card - 1] += 1
    encoding.extend(counts)

    # opponent hand sizes
    for player in clockwise_players[1:]:
        encoding.append(len(state.hands[player]))

    # deal round
    encoding.append(state.deal_round)

    # active player
    encoding.append(state.active_player)

    # skip flag
    encoding.append(int(state.skip_flag))

    return encoding