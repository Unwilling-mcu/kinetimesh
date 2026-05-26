-- KinetiMesh TimescaleDB Schema
-- Hypertables for time-series energy data

CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Harvester node power readings
CREATE TABLE IF NOT EXISTS node_telemetry (
    ts          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    node_id     TEXT NOT NULL,
    node_type   TEXT NOT NULL,
    power_kw    DOUBLE PRECISION,
    voltage_mv  DOUBLE PRECISION,
    battery_soc DOUBLE PRECISION,
    fl_mae      DOUBLE PRECISION,
    predicted_kw DOUBLE PRECISION
);
SELECT create_hypertable('node_telemetry', 'ts', if_not_exists => TRUE);
CREATE INDEX ON node_telemetry (node_id, ts DESC);

-- FL training rounds
CREATE TABLE IF NOT EXISTS fl_rounds (
    ts          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    round_num   INTEGER,
    global_mae  DOUBLE PRECISION,
    fedprox_mae DOUBLE PRECISION,
    fedavg_mae  DOUBLE PRECISION,
    n_clients   INTEGER
);
SELECT create_hypertable('fl_rounds', 'ts', if_not_exists => TRUE);

-- RL agent steps
CREATE TABLE IF NOT EXISTS rl_steps (
    ts             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    step_num       BIGINT,
    episode_reward DOUBLE PRECISION,
    policy_loss    DOUBLE PRECISION,
    green_util_pct DOUBLE PRECISION,
    grid_import_pct DOUBLE PRECISION,
    zone_allocs    JSONB
);
SELECT create_hypertable('rl_steps', 'ts', if_not_exists => TRUE);

-- Blockchain energy credit transactions
CREATE TABLE IF NOT EXISTS credit_transactions (
    ts          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tx_hash     TEXT UNIQUE,
    from_zone   TEXT,
    to_zone     TEXT,
    credits_kwh DOUBLE PRECISION,
    value_inr   DOUBLE PRECISION,
    block_num   INTEGER,
    finality_s  DOUBLE PRECISION
);
SELECT create_hypertable('credit_transactions', 'ts', if_not_exists => TRUE);

-- Anomaly events
CREATE TABLE IF NOT EXISTS anomaly_events (
    ts          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    node_id     TEXT,
    score       DOUBLE PRECISION,
    anomaly_type TEXT,
    resolved_ts TIMESTAMPTZ
);
SELECT create_hypertable('anomaly_events', 'ts', if_not_exists => TRUE);

-- Continuous aggregates for dashboard
CREATE MATERIALIZED VIEW IF NOT EXISTS node_power_1min
WITH (timescaledb.continuous) AS
SELECT time_bucket('1 minute', ts) AS bucket,
       node_id, node_type,
       AVG(power_kw) AS avg_kw, MAX(power_kw) AS max_kw,
       COUNT(*) AS readings
FROM node_telemetry
GROUP BY bucket, node_id, node_type
WITH NO DATA;

COMMENT ON TABLE node_telemetry IS 'KinetiMesh harvester node time-series readings';
COMMENT ON TABLE fl_rounds IS 'FedProx federated learning training rounds';
COMMENT ON TABLE rl_steps IS 'PPO reinforcement learning training steps';
COMMENT ON TABLE credit_transactions IS 'Hyperledger Fabric energy credit ledger';
