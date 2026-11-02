"""B2 — ZIP-centroid distance heuristic (Chicago area).

Distance is a heuristic for *ranking* only; the supplier's "delivers to ZIP" answer is authoritative.
"""

from __future__ import annotations

import math

# Approximate ZIP centroids (lat, lon). Directory ZIPs first, then nearby ZIPs a patient might give.
ZIP_CENTROIDS: dict[str, tuple[float, float]] = {
    # directory ZIPs (data/suppliers.csv)
    "60614": (41.9227, -87.6533),  # Lincoln Park
    "60654": (41.8923, -87.6373),  # River North
    "60618": (41.9464, -87.7042),  # Avondale / North Center
    "60640": (41.9719, -87.6624),  # Uptown
    "60605": (41.8676, -87.6174),  # South Loop
    "60608": (41.8469, -87.6705),  # Pilsen
    "60657": (41.9399, -87.6528),  # Lakeview
    "60714": (42.0294, -87.8120),  # Niles
    "60201": (42.0546, -87.6944),  # Evanston
    # nearby patient ZIPs
    "60610": (41.9035, -87.6337),  # Old Town / Gold Coast
    "60611": (41.8948, -87.6205),  # Streeterville
    "60613": (41.9543, -87.6575),  # Lakeview North / Wrigleyville
    "60622": (41.9024, -87.6829),  # Wicker Park
    "60625": (41.9703, -87.7044),  # Lincoln Square
    "60647": (41.9209, -87.7017),  # Logan Square
    "60616": (41.8424, -87.6309),  # Near South Side
    "60660": (41.9909, -87.6629),  # Edgewater
}

_EARTH_RADIUS_MI = 3958.8


def _haversine_mi(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * _EARTH_RADIUS_MI * math.asin(math.sqrt(h))


def _zip5(z: str) -> str:
    return str(z).strip()[:5]


def distance_mi(zip_a: str, zip_b: str) -> float:
    """Miles between two ZIP centroids; unknown ZIP → abs(int(a) - int(b)) * 0.5 (rounded to 0.1 mi)."""
    a, b = _zip5(zip_a), _zip5(zip_b)
    if a == b:
        return 0.0
    if a in ZIP_CENTROIDS and b in ZIP_CENTROIDS:
        return round(_haversine_mi(ZIP_CENTROIDS[a], ZIP_CENTROIDS[b]), 1)
    try:
        return round(abs(int(a) - int(b)) * 0.5, 1)
    except ValueError:
        return float("inf")
