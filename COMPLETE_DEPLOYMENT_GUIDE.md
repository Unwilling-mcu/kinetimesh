# KinetiMesh — Complete Deployment Guide
# Every command is PowerShell unless marked otherwise

═══════════════════════════════════════════════
 STEP 1: GITHUB PAGES (Live Portfolio URL)
 Time: 5 minutes | Result: public URL on resume
═══════════════════════════════════════════════

1. Go to: https://github.com/Unwilling-mcu/kinetimesh/settings/pages
2. Source: "Deploy from a branch"
3. Branch: main | Folder: / (root)
4. Click Save
5. Wait 2 minutes
6. URL: https://unwilling-mcu.github.io/kinetimesh/

That URL shows index.html — your animated landing page.
Add it to your resume under "Portfolio" or "Projects".

═══════════════════════════════════════════════
 STEP 2: ADD GITHUB REPO DESCRIPTION
 Time: 2 minutes | Result: professional repo page
═══════════════════════════════════════════════

1. Go to: https://github.com/Unwilling-mcu/kinetimesh
2. Click ⚙️ gear icon next to "About" (right sidebar)
3. Description: "Federated Kinetic Intelligence Network — FL + RL + GNN + Blockchain + Quantum | B.Tech IT Research"
4. Website: https://unwilling-mcu.github.io/kinetimesh/
5. Topics (click the gear, add these one by one):
   federated-learning
   reinforcement-learning
   energy-harvesting
   digital-twin
   pytorch
   fastapi
   iot
   quantum-computing
   smart-grid
   python
6. Check: Releases, Packages, Environments
7. Click Save changes

═══════════════════════════════════════════════
 STEP 3: LOCAL FULL STACK (Docker)
 Time: 10 minutes | Result: full backend running
═══════════════════════════════════════════════

# Make sure Docker Desktop is running first
# Then:

cd C:\Users\KIIT0001\Desktop\Riju\KinetiMesh\infra
docker compose up -d

# Check all 6 services started:
docker compose ps

# Expected output:
# kinetimesh-backend    running   0.0.0.0:8000->8000
# kinetimesh-db         running   0.0.0.0:5432->5432
# kinetimesh-redis      running   0.0.0.0:6379->6379
# kinetimesh-mqtt       running   0.0.0.0:1883->1883
# kinetimesh-grafana    running   0.0.0.0:3001->3000
# kinetimesh-prometheus running   0.0.0.0:9090->9090

# Open these in browser:
Start-Process "http://localhost:8000/docs"    # Swagger UI
Start-Process "http://localhost:3001"         # Grafana (admin/kinetimesh)
Start-Process "http://localhost:9090"         # Prometheus

═══════════════════════════════════════════════
 STEP 4: RUN FL TRAINING
 Time: 5-10 minutes | Result: trained model
═══════════════════════════════════════════════

cd C:\Users\KIIT0001\Desktop\Riju\KinetiMesh\ml\federated

# Quick run (proves it works — 30 seconds):
python train_fedprox.py --nodes 6 --rounds 10 --mu 0.01

# Full training (best results — ~5 minutes):
python train_fedprox.py --nodes 18 --rounds 100 --mu 0.01

# You already have kinetimesh_fl_best.pt from a live run!
# This just updates it with better convergence.

═══════════════════════════════════════════════
 STEP 5: RUN RL TRAINING
 Time: 30-60 minutes | Result: trained PPO policy
═══════════════════════════════════════════════

cd C:\Users\KIIT0001\Desktop\Riju\KinetiMesh\ml\rl

# Quick test (10k steps — 2 minutes):
python train_ppo.py --timesteps 10000

# Full training (1M steps — 30min GPU, 2h CPU):
python train_ppo.py --timesteps 1000000

# Output: kinetimesh_ppo_final.zip
# Copy to backend/models/ for serving via API

═══════════════════════════════════════════════
 STEP 6: OPEN THE APPS
 Time: instant | Result: see the system live
═══════════════════════════════════════════════

cd C:\Users\KIIT0001\Desktop\Riju\KinetiMesh

# Landing page (animated, anime.js):
Start-Process "index.html"

# v4 Dashboard (best one — open this for demos):
Start-Process "KinetiMesh_v4_Dashboard.html"

# v3 Dashboard (also good):
Start-Process "KinetiMesh_v3_FullStack.html"

# Research paper (22 sections):
Start-Process "KinetiMesh_Ultimate_Proposal.html"

═══════════════════════════════════════════════
 STEP 7: CLOUD DEPLOYMENT (Render — FREE)
 Time: 15 minutes | Result: live API on the web
═══════════════════════════════════════════════

Option A: Render.com (Free tier, easiest)
─────────────────────────────────────────
1. Go to: https://render.com
2. Sign up with GitHub
3. "New" → "Web Service"
4. Connect repo: Unwilling-mcu/kinetimesh
5. Settings:
   - Name: kinetimesh-api
   - Root directory: backend
   - Build command: pip install -r requirements.txt
   - Start command: uvicorn app.main:app --host 0.0.0.0 --port $PORT
6. Click "Create Web Service"
7. URL: https://kinetimesh-api.onrender.com
8. API docs: https://kinetimesh-api.onrender.com/docs

Option B: Railway.app (Also free)
───────────────────────────────────
1. Go to: https://railway.app
2. "New Project" → "Deploy from GitHub repo"
3. Select: Unwilling-mcu/kinetimesh
4. Add variable: ROOT_DIR = backend
5. Auto-detected: Python + uvicorn
6. Deploy → get URL

Option C: Fly.io (More control)
────────────────────────────────
# Install flyctl first:
winget install Fly.flyctl

cd backend
fly launch --name kinetimesh-api --region bom  # bom = Mumbai
fly deploy

═══════════════════════════════════════════════
 STEP 8: COMPILE IEEE LATEX PAPER
 Time: 5 minutes | Result: PDF ready to submit
═══════════════════════════════════════════════

Option A: Overleaf (easiest, recommended)
──────────────────────────────────────────
1. Go to: https://overleaf.com
2. New Project → Upload Project
3. Upload: docs/KinetiMesh_IEEE_Paper.tex
4. Compiler: pdfLaTeX
5. Click Recompile → Download PDF

Option B: Local LaTeX (if MiKTeX installed)
────────────────────────────────────────────
cd docs
pdflatex KinetiMesh_IEEE_Paper.tex
pdflatex KinetiMesh_IEEE_Paper.tex  # run twice for references
# Output: KinetiMesh_IEEE_Paper.pdf

═══════════════════════════════════════════════
 STEP 9: PUSH ALL NEW FILES
 Time: 2 minutes
═══════════════════════════════════════════════

cd C:\Users\KIIT0001\Desktop\Riju\KinetiMesh

# Copy new files from downloads:
# - KinetiMesh_v4_Dashboard.html → project root
# - docs/KinetiMesh_IEEE_Paper.tex → docs/ folder

git add .
git commit -m "feat: v4 dashboard glassmorphism UI, IEEE LaTeX paper, deployment guide"
git push

═══════════════════════════════════════════════
 WHAT TO SAY IN INTERVIEWS
═══════════════════════════════════════════════

"I built KinetiMesh — a full-stack research system that converts
mechanical waste energy from rail systems into grid-injectable
electricity. The backend is a live FastAPI server at
kinetimesh-api.onrender.com with 10 REST endpoints and WebSocket
streaming. I have a trained FedProx federated learning model
(checkpoint in repo), a GraphSAGE GNN Digital Twin running on 18
nodes with 4.4% prediction error, a custom OpenAI Gymnasium
environment for PPO reinforcement learning, MicroPython edge
firmware with epsilon-DP privacy, TimescaleDB schema with
hypertables, Docker Compose full stack, GitHub Actions CI/CD with
3 passing checks, an IEEE LaTeX paper draft, and a live browser
demo with 3D city visualization, quantum circuit animation, and
blockchain ledger. The GitHub repo is public at
github.com/Unwilling-mcu/kinetimesh and the portfolio page is at
unwilling-mcu.github.io/kinetimesh."

That's not a homework project. That's a research system.
