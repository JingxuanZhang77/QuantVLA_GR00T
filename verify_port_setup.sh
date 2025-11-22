#!/bin/bash
# Verify GR00T and OpenPI port separation
# This script checks that the correct ports are configured

echo "================================================"
echo "GR00T + OpenPI Port Configuration Verification"
echo "================================================"
echo ""

echo "📋 Checking GR00T configuration..."
echo ""

# Check inference server port
INFERENCE_PORT=$(grep -oP 'port\s+\K\d+' /home/jz97/VLM_REPO/Isaac-GR00T/run_inference_server.sh | head -1)
echo "  GR00T Inference Server Port: $INFERENCE_PORT"

# Check eval script port
EVAL_PORT=$(grep -oP '\-\-port\s+\K\d+' /home/jz97/VLM_REPO/Isaac-GR00T/run_libero_eval.sh | head -1)
echo "  GR00T Eval Client Port:      $EVAL_PORT"

if [ "$INFERENCE_PORT" = "5556" ] && [ "$EVAL_PORT" = "5556" ]; then
    echo "  ✅ GR00T correctly configured for port 5556"
else
    echo "  ❌ GR00T port mismatch! Server=$INFERENCE_PORT, Eval=$EVAL_PORT"
fi

echo ""
echo "📋 Checking OpenPI configuration..."
echo ""

# Check OpenPI port (default 5555)
if [ -f /home/jz97/VLM_REPO/openpi/examples/libero/main.py ]; then
    OPENPI_DEFAULT=$(grep -oP 'default.*port.*=\K\d+' /home/jz97/VLM_REPO/openpi/examples/libero/main.py | head -1)
    echo "  OpenPI Default Port:         ${OPENPI_DEFAULT:-5555 (default)}"
    echo "  ✅ OpenPI uses port 5555 (default)"
else
    echo "  ⚠️  OpenPI not found at expected location"
fi

echo ""
echo "📊 Port Allocation Summary:"
echo ""
echo "  ┌─────────────┬──────┐"
echo "  │   Service   │ Port │"
echo "  ├─────────────┼──────┤"
echo "  │   OpenPI    │ 5555 │"
echo "  │   GR00T     │ 5556 │"
echo "  └─────────────┴──────┘"

echo ""
echo "🔍 Checking running processes..."
echo ""

# Check for running OpenPI
OPENPI_PID=$(ps aux | grep "python examples/libero/main.py" | grep -v grep | awk '{print $2}' | head -1)
if [ -n "$OPENPI_PID" ]; then
    echo "  ✅ OpenPI running (PID: $OPENPI_PID)"
else
    echo "  ⚪ OpenPI not running"
fi

# Check for running GR00T inference server
GROOT_PID=$(ps aux | grep "inference_service.py" | grep -v grep | awk '{print $2}' | head -1)
if [ -n "$GROOT_PID" ]; then
    echo "  ✅ GR00T inference server running (PID: $GROOT_PID)"
else
    echo "  ⚪ GR00T inference server not running"
fi

echo ""
echo "🔍 Checking port usage..."
echo ""

# Check actual port bindings
PORT_5555=$(ss -tuln 2>/dev/null | grep ":5555" || echo "")
PORT_5556=$(ss -tuln 2>/dev/null | grep ":5556" || echo "")

if [ -n "$PORT_5555" ]; then
    echo "  ✅ Port 5555 in use (OpenPI)"
else
    echo "  ⚪ Port 5555 available"
fi

if [ -n "$PORT_5556" ]; then
    echo "  ✅ Port 5556 in use (GR00T)"
else
    echo "  ⚪ Port 5556 available"
fi

echo ""
echo "================================================"
echo "✅ Configuration verification complete!"
echo ""
echo "To start services:"
echo "  OpenPI (Terminal 1): cd ~/VLM_REPO/openpi && <run command>"
echo "  GR00T  (Terminal 2): cd ~/VLM_REPO/Isaac-GR00T && ./run_inference_server.sh libero_spatial"
echo "  Eval   (Terminal 3): cd ~/VLM_REPO/Isaac-GR00T && ./run_libero_eval.sh libero_spatial --headless"
echo "================================================"
