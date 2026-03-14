#!/bin/bash
# 在 Fluidstack H100 上執行
# 用法：bash backup_remote.sh

set -e

PDD_DIR="/home/wei/PowerDecode"
TIMESTAMP=$(date +%Y%m%d_%H%M)
BACKUP_NAME="powerdecode_demo_${TIMESTAMP}.db"

echo "=== PowerDecode Demo Backup ==="
echo "時間：$TIMESTAMP"

# 1. 確認 DB 有數據
ROW_COUNT=$(sqlite3 $PDD_DIR/data/powerdecode.db "SELECT COUNT(*) FROM requests;")
echo "DB 筆數：$ROW_COUNT"

if [ "$ROW_COUNT" -lt 30 ]; then
  echo "❌ 數據不足 30 筆，請先跑 seed_demo_data.py"
  exit 1
fi

# 2. 確認有 anomaly 數據
ANOMALY_COUNT=$(sqlite3 $PDD_DIR/data/powerdecode.db "SELECT COUNT(*) FROM requests WHERE anomaly_flag=1;")
echo "Anomaly 筆數：$ANOMALY_COUNT"

if [ "$ANOMALY_COUNT" -lt 1 ]; then
  echo "⚠️  警告：沒有 anomaly_flag=1 的數據，dashboard 標紅不會出現"
fi

# 3. 備份 DB
cp $PDD_DIR/data/powerdecode.db $PDD_DIR/data/$BACKUP_NAME
echo "✓ 備份完成：$PDD_DIR/data/$BACKUP_NAME"

# 4. 印出 SCP 指令讓你複製貼上
echo ""
echo "=== 在本機執行以下指令 ==="
echo "scp user@\$(curl -s ifconfig.me):$PDD_DIR/data/$BACKUP_NAME ~/pdd_demo_backup.db"
echo ""
echo "=== 備份摘要 ==="
sqlite3 $PDD_DIR/data/powerdecode.db "
SELECT
  COUNT(*) as total_requests,
  ROUND(SUM(energy_joules),1) as total_energy_j,
  COUNT(CASE WHEN anomaly_flag=1 THEN 1 END) as anomaly_count,
  ROUND(MAX(cost_usd),10) as max_cost,
  ROUND(MIN(cost_usd),10) as min_cost
FROM requests;
"
