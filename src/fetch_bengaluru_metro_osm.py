"""
Fetch Namma Metro (subway) stop coordinates from OpenStreetMap and write
``bengaluru_metro_stations_osm.csv`` in the project root.

Run once if the CSV is missing:
    python src/fetch_bengaluru_metro_osm.py
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "bengaluru_metro_stations_osm.csv"
OVERPASS = "https://overpass-api.de/api/interpreter"
S, W, N, E = 12.7, 77.3, 13.2, 77.9

QUERY = f"""
[out:json][timeout:180];
(
  node["station"="subway"]({S},{W},{N},{E});
  node["railway"="station"]["station"="subway"]({S},{W},{N},{E});
);
out;
"""


def main() -> None:
    req = urllib.request.Request(
        OVERPASS,
        data=QUERY.encode(),
        headers={"User-Agent": "HousePriceResearch/1.0"},
    )
    with urllib.request.urlopen(req, timeout=200) as resp:
        data = json.load(resp)
    rows = []
    for e in data.get("elements", []):
        if e.get("type") != "node":
            continue
        tags = e.get("tags", {})
        name = tags.get("name:en") or tags.get("name") or ""
        if "proposed" in name.lower():
            continue
        rows.append({"osm_id": e["id"], "lat": e["lat"], "lon": e["lon"], "name": name})
    import pandas as pd

    df = pd.DataFrame(rows).drop_duplicates(subset=["lat", "lon"])
    df.to_csv(OUT, index=False)
    print(f"Wrote {OUT} ({len(df)} stops)")


if __name__ == "__main__":
    main()
