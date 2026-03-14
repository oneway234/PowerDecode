#!/bin/bash
# Clear DB, restart proxy + dashboard, and re-seed demo data
# Usage: bash scripts/reseed.sh

set -e

PDD_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DB_PATH="$PDD_DIR/data/powerdecode.db"
PROXY_PORT="${PDD_PROXY_PORT:-8001}"
DASH_PORT="${PDD_DASH_PORT:-8501}"

echo "=== PowerDecode Reseed ==="

# 1. Clear DB
if [ -f "$DB_PATH" ]; then
    rm "$DB_PATH"
    echo "✓ Deleted $DB_PATH"
else
    echo "  No existing DB found, skipping"
fi

# 2. Restart proxy (must reconnect to fresh DB)
PROXY_PID=$(lsof -ti :$PROXY_PORT 2>/dev/null || true)
if [ -n "$PROXY_PID" ]; then
    kill $PROXY_PID 2>/dev/null || true
    sleep 1
    echo "✓ Stopped old proxy (PID $PROXY_PID)"
fi

echo "  Starting proxy..."
cd "$PDD_DIR"
python3 proxy.py > /tmp/powerdecode_proxy.log 2>&1 &
NEW_PROXY_PID=$!

for i in $(seq 1 30); do
    if curl -s --max-time 2 http://localhost:$PROXY_PORT/health > /dev/null 2>&1; then
        echo "✓ Proxy started (PID $NEW_PROXY_PID)"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "❌ Proxy failed to start. Check /tmp/powerdecode_proxy.log"
        exit 1
    fi
    sleep 1
done

# 3. Restart dashboard (must reconnect to fresh DB)
DASH_PID=$(lsof -ti :$DASH_PORT 2>/dev/null || true)
if [ -n "$DASH_PID" ]; then
    kill $DASH_PID 2>/dev/null || true
    sleep 1
    echo "✓ Stopped old dashboard (PID $DASH_PID)"
fi

echo "  Starting dashboard..."
streamlit run "$PDD_DIR/dashboard.py" --server.port $DASH_PORT --server.headless true > /tmp/powerdecode_dashboard.log 2>&1 &
NEW_DASH_PID=$!
sleep 2

if curl -s --max-time 3 http://localhost:$DASH_PORT > /dev/null 2>&1; then
    echo "✓ Dashboard started (PID $NEW_DASH_PID)"
else
    echo "⚠️ Dashboard may still be starting, check http://localhost:$DASH_PORT"
fi

# 4. Seed demo data
echo ""
echo "=== Seeding demo data ==="
python3 "$PDD_DIR/cluster/seed_demo_data.py"

# 5. Verify
echo ""
echo "=== Verify ==="
ROW_COUNT=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM requests;" 2>/dev/null || echo "0")
ANOMALY_COUNT=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM requests WHERE anomaly_flag >= 1;" 2>/dev/null || echo "0")
echo "Total requests: $ROW_COUNT"
echo "Anomaly requests: $ANOMALY_COUNT"

if [ "$ROW_COUNT" -ge 30 ]; then
    echo "✓ Reseed complete — dashboard: http://localhost:$DASH_PORT"
else
    echo "⚠️ Only $ROW_COUNT requests, expected ≥ 30"
fi
