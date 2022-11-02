from dataclasses import dataclass

from aircraft import Aircraft


@dataclass(eq=True, frozen=True)
class Flight:
    time: int
    origin: str
    destination: str
    airline: str
    aircraft: Aircraft
