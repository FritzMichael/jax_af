
from dataclasses import dataclass

@dataclass(frozen=True)  # frozen makes it hashable for lru_cache
class SarConstants:
    c_0: float = 299792458.0
    b: float = 4.0e8
    tp: float = 9.2e-5
    f_0: float = 9.5e9
    dr: float = 0.0657581761289589