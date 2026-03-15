from typing import List

from state import Zone

class Board:
    JUST_OUT_TILES = {0:0, 1:18, 2:36}
    HOME_ENTRY_TILES = {0:52, 1:16, 2:34}
    MAIN_TRACK_SIZE = 54
    HOME_STRETCH_SIZE = 4
    HOME_STRETCH_STARTING_INDEX = {0:54, 1:58, 2:62}
    HOME_STRETCH_END_INDEX = {0: 57, 1: 61, 2: 65}

    def is_just_out(self, pawn):
        return pawn.zone == Zone.MAIN and pawn.is_newly_deployed

    def is_protected(self, pawn):
        if pawn.zone == Zone.BASE or pawn.zone == Zone.HOME or self.is_just_out(pawn):
            return True
        return False

    def owner_of_tile(self, tile_index):
        if 0 <= tile_index <= 17:
            return 0
        elif 18 <= tile_index <= 35:
            return 1
        elif 36 <= tile_index <= 53:
            return 2
        else:
            raise ValueError('Invalid tile index')

    def home_stretch_entry_tile(self, player):
        return self.HOME_ENTRY_TILES[player]

    def get_path(self, pawn, steps, enter_home: bool = False) -> list[int]:
        path_list = []

        in_home_stretch = pawn.zone == Zone.HOME

        current_index = pawn.index
        for step in range(steps):

            if current_index == self.HOME_ENTRY_TILES[pawn.owner] and enter_home:
                current_index = self.HOME_STRETCH_STARTING_INDEX[pawn.owner]
                in_home_stretch = True
            else:
                current_index = current_index + 1

            if current_index == self.MAIN_TRACK_SIZE and not in_home_stretch:
                current_index = 0

            if in_home_stretch and current_index > self.HOME_STRETCH_END_INDEX[pawn.owner]:
                raise ValueError("Overshoot in home stretch — illegal move")

            path_list.append(current_index)

        return path_list

    def get_path_backward(self, pawn, steps) -> list[int]:
        path_list = []

        if pawn.zone == Zone.HOME:
            raise ValueError("Can't move backward from home stretch")

        current_index = pawn.index
        for step in range(steps):

            current_index = current_index - 1

            if current_index < 0:
                current_index = 53

            path_list.append(current_index)

        return path_list