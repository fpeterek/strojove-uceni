from dataclasses import dataclass


@dataclass(eq=True, frozen=True)
class Aircraft:
    manufacturer: str
    model: str

    def __str__(self):
        return f'{self.manufacturer} {self.model}'
