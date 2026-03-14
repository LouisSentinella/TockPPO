from enum import Enum, auto
from typing import Optional, List
from dataclasses import dataclass

type Card = int

# 1=A, 2-10=2-10, 11=J, 12=Q, 13=K

class Zone(Enum):
    BASE = auto()
    MAIN = auto()
    HOME = auto()

@dataclass
class Pawn:
    owner: int
    zone: Zone
    pawn_id: int
    index: Optional[int] = None
    is_newly_deployed: bool = False

@dataclass
class GameState:
    pawns: List[List[Pawn]]
    hands: List[List[Card]]
    deck: List[Card]
    discard_pile: List[Card]
    active_player: int
    skip_flag: bool
    deal_round: int
    deal_starting_player: int