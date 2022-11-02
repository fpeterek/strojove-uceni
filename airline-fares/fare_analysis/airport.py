from dataclasses import dataclass


@dataclass(eq=True, frozen=True)
class Airport:
    icao: str
    iata: str
    name: str
    country: str
