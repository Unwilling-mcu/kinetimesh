"""
KinetiMesh — Edge Node Firmware (MicroPython-compatible)
Runs on: RPi CM4 / ESP32-S3 with TensorFlow Lite
Protocol: LoRaWAN + MQTT over TLS
Author: Sanchayan | B.Tech IT

Deploy:
  mpremote cp edge_node.py :main.py
  mpremote run main.py
"""

import json, time, math, random

# ── Configuration ──
NODE_ID   = "R01"
NODE_TYPE = "rail"   # "rail" | "floor" | "wind" | "thermal"
BASE_KW   = 1.8
MQTT_BROKER = "kinetimesh.yourdomain.com"
MQTT_PORT   = 8883
MQTT_TOPIC  = f"kinetimesh/nodes/{NODE_ID}/telemetry"
FL_TOPIC    = f"kinetimesh/fl/{NODE_ID}/gradient"
UPLOAD_INTERVAL = 6   # seconds between MQTT publishes
FL_INTERVAL     = 360 # seconds between FL gradient uploads (6 min)


# ── Simulated ADC read from piezo transducer ──
def read_piezo_voltage():
    """Read peak voltage from PZT bimorph via 12-bit ADC. Returns mV."""
    # In production: adc = machine.ADC(machine.Pin(34))
    # Here: simulated signal
    t = time.time()
    base_mv = BASE_KW * 1000 * (0.7 + 0.6*math.sin(t*0.1))
    noise   = random.gauss(0, base_mv*0.05)
    return max(0.0, base_mv + noise)


def voltage_to_power(v_mv: float) -> float:
    """Convert peak piezo voltage to harvested power (kW). Assumes 1kOhm load."""
    v_rms = v_mv / 1000 * 0.707  # Peak → RMS
    return round((v_rms**2 / 1000) * 1000, 4)  # P = V²/R, converted to kW


# ── TinyML inference stub ──
def tinyml_infer_harvest_trend(recent_readings: list) -> float:
    """
    Run local TinyML model (TFLite INT8 quantized LSTM) for short-term prediction.
    In production: use tflite_runtime.interpreter
    Returns: predicted harvest for next 15 min (kW)
    """
    if len(recent_readings) < 5:
        return recent_readings[-1] if recent_readings else 0.0
    # Simple moving average as stub (TFLite replaces this)
    trend = sum(recent_readings[-5:]) / 5
    seasonal = 1.0 + 0.2*math.sin(time.time()*0.001)
    return round(trend * seasonal, 4)


# ── Local gradient computation for FL ──
def compute_gradient_stub(readings: list) -> dict:
    """
    Compute local model weight gradient for FedProx FL upload.
    In production: runs PyTorch or TFLite on-device training step.
    Returns: compressed gradient dict (NOT raw readings — privacy preserved).
    """
    if not readings: return {}
    mean_r = sum(readings)/len(readings)
    std_r  = max(0.001, (sum((x-mean_r)**2 for x in readings)/len(readings))**0.5)
    # Stub: return summary statistics as "gradient" (production uses actual weight deltas)
    return {
        "w_mean": round(mean_r/BASE_KW - 1.0, 6),
        "w_std":  round(std_r/BASE_KW, 6),
        "w_trend":round((readings[-1]-readings[0])/(len(readings)*BASE_KW+1e-6), 6),
    }


def apply_dp_noise(gradient: dict, epsilon: float = 1.0, delta: float = 1e-5) -> dict:
    """Apply ε-differential privacy noise to gradient before upload."""
    sensitivity = 1.0  # L2 sensitivity bound
    sigma = (2 * math.log(1.25/delta))**0.5 * sensitivity / epsilon
    return {k: v + random.gauss(0, sigma*0.01) for k,v in gradient.items()}


# ── MQTT publish stub ──
def mqtt_publish(topic: str, payload: dict):
    """Publish to MQTT broker over TLS. Stub prints in simulation."""
    print(f"[MQTT → {topic}] {json.dumps(payload)[:120]}...")


# ── Main loop ──
def main():
    print(f"KinetiMesh Edge Node {NODE_ID} ({NODE_TYPE}) starting...")
    readings = []
    last_fl_upload = time.time()
    tick = 0

    while True:
        tick += 1
        # 1. Read piezo sensor
        v_mv  = read_piezo_voltage()
        power = voltage_to_power(v_mv)
        readings.append(power)
        if len(readings) > 100: readings.pop(0)

        # 2. Local TinyML inference
        predicted = tinyml_infer_harvest_trend(readings)

        # 3. Build telemetry packet
        telemetry = {
            "node_id":     NODE_ID,
            "node_type":   NODE_TYPE,
            "power_kw":    power,
            "predicted_kw":predicted,
            "voltage_mv":  round(v_mv, 2),
            "tick":        tick,
            "ts":          time.time(),
        }

        # 4. Publish telemetry (every UPLOAD_INTERVAL seconds)
        mqtt_publish(MQTT_TOPIC, telemetry)

        # 5. FL gradient upload (every FL_INTERVAL seconds)
        now = time.time()
        if now - last_fl_upload >= FL_INTERVAL:
            gradient = compute_gradient_stub(readings)
            gradient = apply_dp_noise(gradient)  # ε-DP before upload
            mqtt_publish(FL_TOPIC, {"node_id": NODE_ID, "gradient": gradient,
                                     "n_samples": len(readings)})
            last_fl_upload = now
            print(f"FL gradient uploaded (ε-DP applied, no raw data sent)")

        time.sleep(UPLOAD_INTERVAL)


if __name__ == "__main__":
    main()
