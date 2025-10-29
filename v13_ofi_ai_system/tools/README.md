# harvestd - Quick Reference

24x7 OFI+CVD data collection daemon with self-healing capabilities.

## Installation

### Prerequisites
- Python 3.11+
- Required packages: asyncio, websockets, pandas, pyarrow, numpy

### Quick Test (Manual)
```bash
cd v13_ofi_ai_system
python tools/harvestd.py
```

Access UI: http://localhost:8088/

## Features

✅ **Automatic Recovery**
- Subprocess crashes → exponential backoff restart
- Network interruptions handled gracefully
- System-level restart (systemd/Docker)

✅ **Data Quality Monitoring**
- Hourly validation of collected data
- DoD checks: completeness, deduplication, latency, 2×2 scene coverage
- Automatic alerts on repeated failures

✅ **Minimal Web UI**
- Status dashboard with 5s auto-refresh
- Health check endpoint
- Recent logs viewer
- Latest DQ report viewer
- Prometheus metrics export

## Endpoints

| URL | Description |
|-----|-------------|
| `http://localhost:8088/` | Status page (HTML) |
| `http://localhost:8088/health` | Health check (plain text) |
| `http://localhost:8088/logs` | Recent log lines |
| `http://localhost:8088/dq` | Latest DQ report (JSON) |
| `http://localhost:8088/metrics` | Prometheus metrics |

## Configuration

### Environment Variables

```bash
# Daemon settings
export HARVESTD_PORT=8088
export VALIDATE_INTERVAL_MIN=60
export DQ_FAIL_MAX_TOL=2

# Data collection (from Task 1.3.1)
export SYMBOLS="BTCUSDT,ETHUSDT,BNBUSDT,XRPUSDT,ADAUSDT,SOLUSDT"
export RUN_HOURS=72
export PARQUET_ROTATE_SEC=60

# Scenario labels
export SCENARIO_SCHEME=regime2x2
export WIN_SECS=300
export ACTIVE_TPS=2.0
```

See `deploy/README.md` for complete configuration reference.

## Deployment

### Docker Compose
```bash
cd v13_ofi_ai_system
docker-compose -f deploy/docker-compose.yml up -d
```

### Systemd
```bash
sudo cp deploy/harvestd.service /etc/systemd/system/
sudo systemctl enable harvestd
sudo systemctl start harvestd
```

## Monitoring

### Check Status
```bash
curl http://localhost:8088/health
```

### View Logs
```bash
# Systemd
sudo journalctl -u harvestd -f

# Docker
docker-compose logs -f harvestd
```

### View Data Quality
```bash
curl http://localhost:8088/dq | jq
```

## Data Flow

```
Binance Futures WebSocket
        ↓
run_success_harvest.py (subprocess)
        ↓
data/ofi_cvd/ (Parquet files)
        ↓
validate_ofi_cvd_harvest.py (hourly)
        ↓
artifacts/dq_reports/ (DQ reports)
```

## Output Structure

```
data/
└── ofi_cvd/
    └── date=YYYY-MM-DD/
        └── symbol=<SYMBOL>/
            ├── kind=prices/
            ├── kind=ofi/
            ├── kind=cvd/
            ├── kind=fusion/
            └── kind=events/

artifacts/
├── run_logs/          # Daemon logs
└── dq_reports/        # DQ validation reports
    ├── dq_YYYYMMDD_HHMM.json
    └── slices_manifest_YYYYMMDD_HH.json
```

## Troubleshooting

### Service won't start
```bash
# Check logs
tail -f artifacts/run_logs/harvestd.log

# Check Python path
which python3
python3 --version
```

### Data collection failing
```bash
# Test harvester script manually
python examples/run_success_harvest.py --help

# Check environment variables
env | grep SYMBOLS
env | grep WSS_
```

### Data quality checks failing
```bash
# Run validator manually
python scripts/validate_ofi_cvd_harvest.py \
    --base-dir data/ofi_cvd \
    --output-dir artifacts/dq_reports
```

## Integration with Existing Scripts

**No changes required** to existing scripts:
- `examples/run_success_harvest.py` - Runs as subprocess
- `scripts/validate_ofi_cvd_harvest.py` - Runs as scheduled task

All existing features preserved:
- 2×2 scenario labels (A_H, A_L, Q_H, Q_L)
- Session labels (Tokyo, London, NY)
- Fee tier labels
- DoD validation checks
- Data quality thresholds

## Next Steps

1. **Deploy**: Choose Docker or systemd deployment
2. **Monitor**: Access http://localhost:8088/ for status
3. **Collect**: Data accumulates in `data/ofi_cvd/`
4. **Validate**: Hourly DQ reports in `artifacts/dq_reports/`
5. **Integrate**: Use collected data for downstream analysis (Task 1.3.2)

## Support

- Full documentation: `deploy/README.md`
- Task spec: `TASKS/Stage1_真实OFI+CVD核心/✅Task_1.3.1_收集历史OFI+CVD数据.md`
- Validation: `scripts/validate_ofi_cvd_harvest.py`
