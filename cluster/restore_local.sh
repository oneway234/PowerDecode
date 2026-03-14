#!/bin/bash
# 在本機執行
# 用法：bash restore_local.sh ~/pdd_demo_backup.db

set -e

BACKUP_FILE=$1
PDD_DIR="$(dirname "$0")/.."
LOCAL_DB="$PDD_DIR/data/powerdecode.db"

echo "=== PowerDecode Demo Restore ==="

# 1. 確認備份檔存在
if [ -z "$BACKUP_FILE" ] || [ ! -f "$BACKUP_FILE" ]; then
  echo "❌ 用法：bash restore_local.sh <backup_file>"
  echo "   例如：bash restore_local.sh ~/pdd_demo_backup.db"
  exit 1
fi

# 2. 如果本機有舊 DB，先備份
if [ -f "$LOCAL_DB" ]; then
  cp "$LOCAL_DB" "${LOCAL_DB}.before_restore"
  echo "✓ 舊 DB 備份至 ${LOCAL_DB}.before_restore"
fi

# 3. Restore
cp "$BACKUP_FILE" "$LOCAL_DB"
echo "✓ Restore 完成"

# 4. 驗證
ROW_COUNT=$(sqlite3 $LOCAL_DB "SELECT COUNT(*) FROM requests;")
ANOMALY_COUNT=$(sqlite3 $LOCAL_DB "SELECT COUNT(*) FROM requests WHERE anomaly_flag=1;")
echo ""
echo "=== Dashboard 檢查清單 ==="
echo "總筆數：$ROW_COUNT"
echo "Anomaly 筆數：$ANOMALY_COUNT"

# 5. 確認 streamlit 可以啟動
echo ""
echo "=== 啟動 Dashboard ==="
echo "執行：streamlit run $PDD_DIR/dashboard.py"
echo ""

# 確認依賴
python3 -c "import streamlit, pandas, plotly, anthropic" 2>/dev/null \
  && echo "✓ 所有依賴已安裝" \
  || echo "❌ 缺少依賴，執行：pip install streamlit pandas plotly anthropic --break-system-packages"

# 確認 ANTHROPIC_API_KEY
if [ -z "$ANTHROPIC_API_KEY" ] && [ ! -f "$PDD_DIR/.env" ]; then
  echo "⚠️  警告：找不到 ANTHROPIC_API_KEY，三維分析功能會失敗"
else
  echo "✓ ANTHROPIC_API_KEY 已設定"
fi

echo ""
echo "=== Restore 完成，可以開始 Demo ==="
