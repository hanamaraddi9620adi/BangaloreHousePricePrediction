import json
import math
import time
import csv
from pathlib import Path
from http.server import BaseHTTPRequestHandler


ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "stacked_model.pkl"
DATA_PATH = ROOT / "house_prices_bangalore.csv"
COMPACT_MODEL_PATH = ROOT / "model.json"

ALL_COORDS = {
    "Bannerghatta Road": (12.9396, 77.6022), "Yelahanka": (13.1007, 77.5963),
    "Sarjapur Road": (12.9246, 77.6383), "Whitefield": (12.9957, 77.7579),
    "Banashankari": (12.9278, 77.5566), "KR Puram": (13.0075, 77.6959),
    "JP Nagar": (12.9097, 77.5866), "Uttarahalli": (12.9056, 77.5455),
    "Rajajinagar": (13.0005, 77.5497), "Marathahalli": (12.9553, 77.6984),
    "Jayanagar": (12.9293, 77.5824), "HSR Layout": (12.9116, 77.6389),
    "BTM Layout": (12.9140, 77.6103), "Koramangala": (12.9357, 77.6241),
    "Indiranagar": (12.9733, 77.6405), "Electronic City": (12.8488, 77.6483),
    "Kengeri": (12.9177, 77.4838), "Bellandur": (12.9320, 77.6843),
    "Malleshwaram": (13.0027, 77.5703), "Hebbal": (13.0382, 77.5919),
}
METRO_STATIONS = {
    "Majestic": (12.9756, 77.5711), "MG Road": (12.9755, 77.6067),
    "Indiranagar Metro": (12.9783, 77.6387), "Baiyappanahalli": (12.9907, 77.6525),
    "RV Road": (12.9215, 77.5802), "Yelachenahalli": (12.8959, 77.5735),
    "Mysore Road": (12.9537, 77.5300), "Peenya": (13.0336, 77.5255),
    "Kengeri Metro": (12.9180, 77.4840), "Whitefield Metro": (12.9965, 77.7610),
}


def haversine(lat1, lon1, lat2, lon2):
    radius = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    value = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(
        math.radians(lat2)
    ) * math.sin(dlon / 2) ** 2
    return radius * 2 * math.atan2(math.sqrt(value), math.sqrt(1 - value))


def nearest_metro(location):
    lat, lon = ALL_COORDS.get(location, (12.9716, 77.5946))
    return min(
        ((name, haversine(lat, lon, mlat, mlon)) for name, (mlat, mlon) in METRO_STATIONS.items()),
        key=lambda item: item[1],
    )


try:
    with COMPACT_MODEL_PATH.open(encoding="utf-8") as model_file:
        MODEL = json.load(model_file)
    MODEL_ERROR = None
except Exception as error:
    MODEL = None
    MODEL_ERROR = str(error)

try:
    with DATA_PATH.open(newline="", encoding="utf-8") as data_file:
        RECORDS = sum(1 for _ in csv.DictReader(data_file))
except Exception:
    RECORDS = 0


def predict(payload):
    if MODEL is None:
        raise RuntimeError(f"Model unavailable: {MODEL_ERROR}")
    bhk = float(payload["bhk"])
    bath = float(payload["bath"])
    area = float(payload["area"])
    age = float(payload["age"])
    location = str(payload["location"])
    furnishing = str(payload["furnishing"])
    property_type = str(payload["property_type"])
    station, distance = nearest_metro(location)
    price_per_sqft = MODEL["location_price_per_sqft"].get(
        location, MODEL["global_price_per_sqft"]
    )
    type_multiplier = MODEL["property_type_multiplier"].get(property_type, 1.0)
    room_multiplier = 1 + max(-0.08, min(0.12, (bhk - 3) * 0.025))
    bath_multiplier = 1 + max(-0.04, min(0.08, (bath - 2) * 0.02))
    age_multiplier = max(0.82, 1 - age * 0.004)
    metro_multiplier = max(0.96, 1.03 - distance * 0.008)
    started = time.perf_counter()
    price = float(area * price_per_sqft * type_multiplier * room_multiplier *
                  bath_multiplier * age_multiplier * metro_multiplier)
    return {
        "price": price,
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "nearest_station": station,
        "distance_km": round(distance, 2),
        "location": location,
        "property_type": property_type,
        "furnishing": furnishing,
    }


class handler(BaseHTTPRequestHandler):
    def _send(self, status, body):
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        self._send(200, {"ok": True})

    def do_GET(self):
        if self.path.split("?")[0] == "/api/health":
            self._send(200, {
                "ok": MODEL is not None,
                "model": "online" if MODEL is not None else "offline",
                "records": RECORDS,
                "error": MODEL_ERROR,
            })
            return
        self._send(404, {"error": "Not found"})

    def do_POST(self):
        if self.path.split("?")[0] != "/api/predict":
            self._send(404, {"error": "Not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            result = predict(payload)
            self._send(200, result)
        except (KeyError, TypeError, ValueError) as error:
            self._send(400, {"error": f"Invalid input: {error}"})
        except Exception as error:
            self._send(500, {"error": str(error)})
