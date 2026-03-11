from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional
from state import Pawn



class ActionType(Enum):
    DEPLOY = auto()
    MOVE = auto()
    SWAP = auto()
    SEVEN = auto()
    DISCARD = auto()

@dataclass
class Action:
    card: int
    action_type: ActionType
    pawn: Optional[Pawn] = None
    path: Optional[list[int]] = None
    enter_home: Optional[bool] = None
    target_pawn: Optional[Pawn] = None
    seven_moves: Optional[list[tuple[Pawn, int]]] = None