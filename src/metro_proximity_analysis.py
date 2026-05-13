"""
Merge house listings with neighbourhood coordinates, attach nearest Namma Metro
stop (OpenStreetMap), and quantify how price relates to metro distance.

Coordinates in ``bengaluru_coords_helper.csv`` are one point per locality
(typically a neighbourhood centroid). Distance to metro is therefore identical
for all listings in the same ``location``; listing-level correlations are
mostly noise. Locality-level summaries (20 areas) give a clearer picture, with
the caveat that sample size is small for formal significance tests.
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HOUSE_CSV = ROOT / "house_prices_bangalore.csv"
COORDS_CSV = ROOT / "bengaluru_coords_helper.csv"
METRO_CSV = ROOT / "bengaluru_metro_stations_osm.csv"
OVERPASS = "https://overpass-api.de/api/interpreter"
BBOX = (12.7, 77.3, 13.2, 77.9)

OVERPASS_QUERY = f"""
[out:json][timeout:180];
(
  node["station"="subway"]({BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]});
  node["railway"="station"]["station"="subway"]({BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]});
);
out;
"""


def haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: float, lon2: float) -> np.ndarray:
    """Vectorized great-circle distance in km (WGS84 sphere)."""
    r = 6371.0088
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def load_or_fetch_metro() -> pd.DataFrame:
    if METRO_CSV.exists():
        return pd.read_csv(METRO_CSV)
    req = urllib.request.Request(
        OVERPASS,
        data=OVERPASS_QUERY.encode(),
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
    df = pd.DataFrame(rows).drop_duplicates(subset=["lat", "lon"])
    df.to_csv(METRO_CSV, index=False)
    return df


def nearest_metro_km(prop_lat: np.ndarray, prop_lon: np.ndarray, metro: pd.DataFrame) -> np.ndarray:
    mlat = metro["lat"].to_numpy()
    mlon = metro["lon"].to_numpy()
    n = len(prop_lat)
    mins = np.full(n, np.inf)
    for i in range(len(mlat)):
        d = haversine_km(prop_lat, prop_lon, float(mlat[i]), float(mlon[i]))
        mins = np.minimum(mins, d)
    return mins


def main() -> None:
    houses = pd.read_csv(HOUSE_CSV)
    coords = pd.read_csv(COORDS_CSV)
    metro = load_or_fetch_metro()

    df = houses.merge(coords, on="location", how="left", validate="m:1")
    miss = df["latitude"].isna().sum()
    if miss:
        raise SystemExit(f"Missing coordinates for {miss} rows after merge.")

    df["price_per_sqft"] = df["price"] / df["area"].replace(0, np.nan)
    lat = df["latitude"].to_numpy(dtype=float)
    lon = df["longitude"].to_numpy(dtype=float)
    df["nearest_metro_km"] = nearest_metro_km(lat, lon, metro)

    # Correlation: expect negative if closer metro -> higher price
    spear = df[["nearest_metro_km", "price", "price_per_sqft", "area", "bhk"]].corr(
        method="spearman"
    )

    bins = [0, 1, 2, 3, 5, 10, np.inf]
    labels = ["0-1 km", "1-2 km", "2-3 km", "3-5 km", "5-10 km", ">10 km"]
    df["dist_band"] = pd.cut(df["nearest_metro_km"], bins=bins, labels=labels, right=False)
    band = (
        df.groupby("dist_band", observed=True)
        .agg(
            n=("price", "size"),
            median_price=("price", "median"),
            median_ppsf=("price_per_sqft", "median"),
            mean_price=("price", "mean"),
        )
        .reset_index()
    )

    print("\n=== Metro proximity vs house prices (Bengaluru) ===\n")
    print(f"Listings: {len(df):,} | Metro stops used: {len(metro):,} (OSM subway stations, proposed omitted)\n")
    print("Spearman correlation (higher magnitude = stronger monotonic link):\n")
    print(spear.loc[["nearest_metro_km"], ["price", "price_per_sqft", "area", "bhk"]].round(3).to_string())
    print("\nMedian price and median price/sqft by distance to nearest metro:\n")
    print(band.to_string(index=False))

    # Simple linear log-price on distance + controls (interpret coef sign)
    from sklearn.linear_model import LinearRegression

    X = df[["nearest_metro_km", "area", "bhk", "bath"]].astype(float)
    y = np.log(df["price"].astype(float))
    model = LinearRegression().fit(X, y)
    names = ["nearest_metro_km", "area", "bhk", "bath"]
    print("\nOLS on log(price): coefficients (holding area, bhk, bath fixed)\n")
    for n, c in zip(names, model.coef_):
        extra = ""
        if n == "nearest_metro_km":
            extra = "  (~{:.2f}% change in price per +1 km, small-delta approx.)".format(100 * (np.exp(c) - 1))
        print(f"  {n:20s}  {c:+.6f}{extra}")
    print(f"  {'intercept':20s}  {model.intercept_:+.6f}")
    r2 = model.score(X, y)
    print(f"\n  R^2 on log(price): {r2:.3f}")

    baseline = band.loc[band["dist_band"] == "0-1 km", "median_ppsf"]
    far = band.loc[band["dist_band"] == "5-10 km", "median_ppsf"]
    if len(baseline) and len(far) and baseline.iloc[0] > 0 and not np.isnan(far.iloc[0]):
        uplift = (baseline.iloc[0] / far.iloc[0] - 1) * 100
        print(f"\nMedian price/sqft: 0-1 km band vs 5-10 km band: ~{uplift:+.1f}% higher near metro (listing medians)\n")

    # Locality-level: one distance per location, avoids duplicate-geo inflation
    by_loc = (
        df.groupby("location", sort=True)
        .agg(
            n=("price", "size"),
            nearest_metro_km=("nearest_metro_km", "first"),
            median_ppsf=("price_per_sqft", "median"),
            median_price=("price", "median"),
        )
        .reset_index()
        .sort_values("nearest_metro_km")
    )
    loc_corr = by_loc[["nearest_metro_km", "median_ppsf"]].corr(method="spearman").iloc[0, 1]
    print("Locality-level (n=20): Spearman(nearest_metro_km, median price/sqft) = {:.3f}".format(loc_corr))
    print("(Negative => pricier sqft tends to occur closer to metro, across named areas.)\n")
    print(by_loc.to_string(index=False))

    out_dir = ROOT
    band.to_csv(out_dir / "metro_distance_bands.csv", index=False)
    by_loc.to_csv(out_dir / "metro_proximity_by_locality.csv", index=False)
    print(f"\nWrote {out_dir / 'metro_distance_bands.csv'} and {out_dir / 'metro_proximity_by_locality.csv'}\n")


if __name__ == "__main__":
    main()
