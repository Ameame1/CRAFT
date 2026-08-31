#!/bin/bash
# ========================================
# Stop All CRAFT Servers
# ========================================

echo "Stopping all CRAFT servers..."

# Find and kill vLLM processes
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true

# Find and kill Python server processes
pkill -f "src/craft/server.py" 2>/dev/null || true
pkill -f "src/judge/server.py" 2>/dev/null || true

echo "All servers stopped."

# Show remaining GPU processes
echo ""
echo "Remaining GPU processes:"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null || echo "No GPU processes found."
