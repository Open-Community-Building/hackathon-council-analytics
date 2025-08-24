#!/usr/bin/env bash
set -euo pipefail

MODEL_LLM="llama3.2"
MODEL_EMBED="mxbai-embed-large"

# 1. Server im Hintergrund starten
ollama serve &

# 2. Kurze Pause, bis der Server bereit ist
sleep 5

# 3. Modell herunterladen
ollama pull "$MODEL_LLM"
ollama pull "$MODEL_EMBED"

# 4. Warte auf den Server‑Prozess (PID 1)
wait
