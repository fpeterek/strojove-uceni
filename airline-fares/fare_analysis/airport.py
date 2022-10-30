from dataclasses import dataclass


@dataclass
class Airport:
    icao: str
    iata: str
    name: str
    country: str
