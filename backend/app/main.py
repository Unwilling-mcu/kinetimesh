"""
KinetiMesh Backend — FastAPI Application
Author: Sanchayan | B.Tech Information Technology
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio, json, logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kinetimesh")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("KinetiMesh starting up...")
    yield
    logger.info("KinetiMesh shutting down")

app = FastAPI(
    title="KinetiMesh API",
    description="Federated Kinetic Intelligence Network — REST + WebSocket API",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []
    async def connect(self, ws: WebSocket):
        await ws.accept(); self.active.append(ws)
    def disconnect(self, ws: WebSocket):
        if ws in self.active: self.active.remove(ws)
    async def broadcast(self, msg: dict):
        dead = []
        for ws in self.active:
            try: await ws.send_text(json.dumps(msg))
            except: dead.append(ws)
        for ws in dead: self.active.remove(ws)

manager = ConnectionManager()

@app.get("/")
async def root():
    return {"service": "KinetiMesh API", "version": "3.0.0",
            "docs": "/docs", "ws": "/ws/v1/stream"}

@app.get("/api/v1/health")
async def health():
    return {"status": "ok", "fl": "running", "rl": "dispatching",
            "blockchain": "synced", "quantum": "idle",
            "ts": datetime.utcnow().isoformat()}

@app.get("/api/v1/nodes")
async def get_nodes():
    import random, math, time
    t = time.time()
    node_defs = [
        ("R01","Rail PEH Alpha","rail",1.8,0),("R02","Rail PEH Beta","rail",2.1,0),
        ("R03","Rail PEH Gamma","rail",1.6,0),("R04","Rail PEH Delta","rail",1.9,0),
        ("R05","Rail PEH Epsilon","rail",1.4,0),("R06","Rail PEH Zeta","rail",2.3,0),
        ("F01","Floor Block A","floor",0.24,1),("F02","Floor Block B","floor",0.18,1),
        ("F03","Floor Block C","floor",0.22,1),("F04","Floor Block D","floor",0.19,1),
        ("F05","Floor Block E","floor",0.21,1),("F06","Floor Block F","floor",0.17,1),
        ("W01","Wind Harvester 1","wind",0.15,2),("W02","Wind Harvester 2","wind",0.12,2),
        ("T01","TEG Segment 1","thermal",0.08,0),("T02","TEG Segment 2","thermal",0.07,0),
        ("S01","Street Tile R1","floor",0.09,2),("S02","Street Tile R2","floor",0.08,2),
    ]
    nodes = []
    for nid,name,ntype,base,zone in node_defs:
        power = base*(0.7+0.6*random.random())
        if ntype=="rail": power *= (0.8+0.4*abs(math.sin(t*0.08+base)))
        nodes.append({"id":nid,"name":name,"type":ntype,"power_kw":round(power,4),
                      "zone":zone,"status":"active","fl_mae":round(0.077+random.random()*0.05,4),
                      "battery_soc":round(45+random.random()*40,1),"ts":datetime.utcnow().isoformat()})
    total = sum(n["power_kw"] for n in nodes)
    return {"nodes":nodes,"count":len(nodes),"total_kw":round(total,4),"ts":datetime.utcnow().isoformat()}

@app.get("/api/v1/nodes/{node_id}/power")
async def get_node_power(node_id: str):
    import random
    return {"id":node_id,"power_kw":round(1.2+random.random()*1.5,4),
            "status":"active","fl_mae":round(0.08+random.random()*0.03,4),
            "last_sync":"<1s","ts":datetime.utcnow().isoformat()}

@app.get("/api/v1/forecast/24h")
async def get_forecast():
    import random, math
    forecast = []
    for h in range(48):
        hour = h/2
        base = 5+3*math.sin((hour-8)*math.pi/12)+2*math.cos((hour-14)*math.pi/8)
        train = 2.5 if int(hour) in [7,8,9,17,18,19] else 0
        val = max(0.5, base+train+random.uniform(-0.5,0.5))
        ci = 0.4+random.random()*0.3
        forecast.append({"hour":f"{int(hour):02d}:{int((hour%1)*60):02d}",
                         "mean_kw":round(val,3),"ci_lower":round(val-ci,3),"ci_upper":round(val+ci,3),
                         "train_event": train>0})
    return {"forecast":forecast,"model":"fedprox-lstm-v2","accuracy":0.923,
            "generated":datetime.utcnow().isoformat()}

@app.get("/api/v1/twin/topology")
async def get_topology():
    import random
    return {"nodes":18,"edges":24,"last_sync":f"{random.randint(20,55)}ms ago",
            "gnn_error_pct":round(2.5+random.random()*2,2),
            "sync_latency_ms":random.randint(25,65),"ts":datetime.utcnow().isoformat()}

@app.post("/api/v1/rl/dispatch")
async def rl_dispatch():
    import random
    allocs = [0.45+random.uniform(-0.05,0.05),0.33+random.uniform(-0.04,0.04),0.22+random.uniform(-0.03,0.03)]
    s = sum(allocs); allocs = [round(a/s,4) for a in allocs]
    return {"action":allocs,"reward":round(60+random.random()*30,2),
            "green_util_pct":round(65+random.random()*20,1),"policy":"ppo-v3",
            "step":random.randint(500000,999999),"ts":datetime.utcnow().isoformat()}

@app.get("/api/v1/quantum/route")
async def quantum_route():
    import random
    adv = round(20+random.random()*20, 2)
    return {"advantage_pct":adv,"qaoa_depth":3,"iterations":80,
            "classical_cost":100,"quantum_cost":round(100-adv,2),
            "status":"converged","ts":datetime.utcnow().isoformat()}

@app.get("/api/v1/blockchain/ledger")
async def get_ledger():
    import random
    txs = []
    zones = ["Zone A", "Zone B", "Zone C"]
    for i in range(5):
        fz = zones[i%3]; tz = zones[(i+1)%3]
        cr = round(0.01+random.random()*0.12,4)
        txs.append({"hash":f"0x{random.randint(0,0xffffff):06x}","from":fz,"to":tz,
                     "credits":cr,"rs":round(cr*8,3),"finality_s":round(1.2+random.random()*1.4,2)})
    return {"transactions":txs,"block":random.randint(100,500),"ts":datetime.utcnow().isoformat()}

@app.post("/api/v1/events/train")
async def inject_train():
    await manager.broadcast({"type":"train_event","intensity":4.8,"duration_s":20,
                              "ts":datetime.utcnow().isoformat()})
    return {"status":"injected","event":"train_pass","intensity":4.8}

@app.post("/api/v1/events/crowd")
async def inject_crowd():
    await manager.broadcast({"type":"crowd_event","intensity":3.2,"duration_s":15,
                              "ts":datetime.utcnow().isoformat()})
    return {"status":"injected","event":"crowd_surge","intensity":3.2}

@app.websocket("/ws/v1/stream")
async def websocket_stream(ws: WebSocket):
    await manager.connect(ws)
    try:
        import random, math, time
        tick = 0
        while True:
            tick += 1
            t = time.time()
            total = sum(1.2*(0.7+0.6*random.random())*(0.8+0.4*abs(math.sin(t*0.08+i))) for i in range(18))
            await ws.send_text(json.dumps({
                "type":"state_snapshot","tick":tick,
                "power_kw":round(total,3),
                "rail_kw":round(total*0.72,3),
                "floor_kw":round(total*0.21,3),
                "fl_round":tick//4,
                "rl_reward":round(-15+80*(min(1,tick/500)**0.5),2),
                "ts":datetime.utcnow().isoformat()
            }))
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        manager.disconnect(ws)
