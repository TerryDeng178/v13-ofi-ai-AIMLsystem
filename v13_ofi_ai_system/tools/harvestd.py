#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
harvestd - OFI+CVD Data Collection Daemon

A 24x7 self-healing service that:
- Runs data collection (run_success_harvest.py) with automatic restart
- Periodic data quality validation (validate_ofi_cvd_harvest.py)
- Provides minimal HTTP UI for monitoring

Usage:
    python tools/harvestd.py
    
Environment Variables:
    HARVESTD_PORT: HTTP UI port (default: 8088)
    VALIDATE_INTERVAL_MIN: Data quality check interval (default: 60)
    RESTART_BACKOFF_MAX_SEC: Max restart backoff (default: 60)
    DQ_FAIL_MAX_TOL: Max consecutive DQ failures before alert (default: 2)
    LOG_TAIL_LINES: Number of log lines to display (default: 200)
"""

import os
import sys
import time
import json
import signal
import threading
import subprocess
import queue
from http.server import BaseHTTPRequestHandler, HTTPServer
from datetime import datetime
from pathlib import Path

# Project root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ART = os.path.join(ROOT, "deploy", "artifacts")
LOG_DIR = os.path.join(ART, "run_logs")
DQ_DIR = os.path.join(ART, "dq_reports")

# Create directories
for d in [LOG_DIR, DQ_DIR]:
    os.makedirs(d, exist_ok=True)

# Configuration
PORT = int(os.getenv("HARVESTD_PORT", "8088"))
VALIDATE_EVERY_MIN = int(os.getenv("VALIDATE_INTERVAL_MIN", "60"))
RESTART_BACKOFF_MAX = int(os.getenv("RESTART_BACKOFF_MAX_SEC", "60"))
DQ_FAIL_MAX_TOL = int(os.getenv("DQ_FAIL_MAX_TOL", "2"))
LOG_TAIL_LINES = int(os.getenv("LOG_TAIL_LINES", "200"))
PY = sys.executable

# Global state
STATE = {
    "harvester_pid": None,
    "restarts": 0,
    "last_start": None,
    "last_exit": None,
    "last_heartbeat": None,
    "last_dq": None,
    "dq_fail_streak": 0,
    "backoff_sec": 1,
    "running": True,
    # 新增：订单簿数据收集状态
    "orderbook_enabled": os.getenv("ENABLE_ORDERBOOK", "1") == "1",
    "orderbook_files_count": 0,
    "orderbook_last_update": None,
    "orderbook_data_size": 0,
}

# Log queue
LOGQ = queue.Queue()

def log(msg):
    """Add message to log queue"""
    s = f"{datetime.utcnow().isoformat()}Z {msg}"
    LOGQ.put(s)
    # Also print to stderr for systemd/cron
    print(s, file=sys.stderr, flush=True)

def tail_log(lines=LOG_TAIL_LINES):
    """Get last N log lines"""
    out = []
    q = queue.Queue()
    while not LOGQ.empty():
        q.put(LOGQ.get())
    while not q.empty():
        item = q.get()
        out.append(item)
        LOGQ.put(item)
    return "\n".join(out[-lines:])

def check_orderbook_status():
    """检查订单簿数据收集状态"""
    try:
        data_dir = os.path.join(ROOT, "data", "ofi_cvd")
        if not os.path.exists(data_dir):
            return
        
        # 统计订单簿文件
        orderbook_files = []
        total_size = 0
        
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if "orderbook" in root and file.endswith(".parquet"):
                    file_path = os.path.join(root, file)
                    orderbook_files.append(file_path)
                    total_size += os.path.getsize(file_path)
        
        # 更新状态
        STATE["orderbook_files_count"] = len(orderbook_files)
        STATE["orderbook_data_size"] = total_size
        
        if orderbook_files:
            # 获取最新文件的修改时间
            latest_file = max(orderbook_files, key=os.path.getmtime)
            STATE["orderbook_last_update"] = datetime.fromtimestamp(
                os.path.getmtime(latest_file)
            ).isoformat() + "Z"
        
    except Exception as e:
        log(f"[orderbook] 检查状态错误: {e}")

def run_harvester():
    """Run harvester process with enhanced automatic restart and recovery"""
    harvest_script = os.path.join(ROOT, "deploy", "run_success_harvest.py")
    consecutive_failures = 0
    max_consecutive_failures = 5
    base_restart_delay = 5
    max_restart_delay = 300  # 5分钟最大延迟
    
    while STATE["running"]:
        STATE["last_start"] = datetime.utcnow().isoformat() + "Z"
        log(f"[harvester] Starting: {harvest_script}")
        
        # Start subprocess with enhanced error handling
        try:
            # 创建日志文件用于调试
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            log_file = os.path.join(LOG_DIR, f"harvester_{timestamp}.log")
            
            # 使用日志文件而不是DEVNULL，便于调试
            with open(log_file, "w", encoding="utf-8") as f:
                p = subprocess.Popen(
                    [PY, harvest_script],
                    cwd=ROOT,
                    stdout=f,
                    stderr=f,
                    env={**os.environ, "PYTHONPATH": ROOT}  # 确保Python路径正确
                )
            STATE["harvester_pid"] = p.pid
            log(f"[harvester] PID: {p.pid}, log_file: {log_file}")
            
            # Enhanced process monitoring with health checks
            last_heartbeat_check = time.time()
            heartbeat_timeout = 30  # 30秒心跳超时
            
            while STATE["running"]:
                rc = p.poll()
                current_time = time.time()
                STATE["last_heartbeat"] = datetime.utcnow().isoformat() + "Z"
                
                if rc is not None:
                    # Process exited
                    log(f"[harvester] Process exited with code: {rc}")
                    break
                
                # 检查进程是否还在运行
                try:
                    # 发送信号0检查进程是否存在
                    p.send_signal(0)
                    last_heartbeat_check = current_time
                except ProcessLookupError:
                    log(f"[harvester] Process {p.pid} not found, assuming crashed")
                    rc = -1
                    break
                
                # 检查心跳超时
                if current_time - last_heartbeat_check > heartbeat_timeout:
                    log(f"[harvester] Heartbeat timeout, killing process")
                    try:
                        p.terminate()
                        p.wait(timeout=10)
                    except:
                        p.kill()
                    rc = -2
                    break
                
                time.sleep(2)
            
            STATE["last_exit"] = f"rc={rc}"
            STATE["restarts"] += 1
            
            # 计算重启延迟（指数退避）
            if rc != 0:
                consecutive_failures += 1
                restart_delay = min(base_restart_delay * (2 ** min(consecutive_failures, 6)), max_restart_delay)
                log(f"[harvester] Abnormal exit (rc={rc}), consecutive failures: {consecutive_failures}")
                log(f"[harvester] Restarting in {restart_delay}s...")
                time.sleep(restart_delay)
            else:
                consecutive_failures = 0
                log(f"[harvester] Exited normally (rc=0), restarting in 5s...")
                # 修复：即使正常退出也要重启，因为采集器应该24x7运行
                time.sleep(5)  # 短暂等待后重启
                
        except Exception as e:
            log(f"[harvester] Exception: {e}")
            consecutive_failures += 1
            restart_delay = min(base_restart_delay * (2 ** min(consecutive_failures, 6)), max_restart_delay)
            log(f"[harvester] Exception occurred, restarting in {restart_delay}s...")
            time.sleep(restart_delay)

def run_validator():
    """Run periodic data quality validation"""
    script = os.path.join(ROOT, "scripts", "validate_ofi_cvd_harvest.py")
    base_dir = os.path.join(ROOT, "deploy", "data", "ofi_cvd")
    
    while STATE["running"]:
        t0 = time.time()
        try:
            log("[dq] Starting validation job")
            
            # Run validation
            rc = subprocess.call(
                [PY, script, "--base-dir", base_dir, "--output-dir", DQ_DIR],
                cwd=ROOT,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            # Read latest DQ report
            dq_files = list(Path(DQ_DIR).glob("dq_*.json"))
            if dq_files:
                latest = max(dq_files, key=lambda p: p.stat().st_mtime)
                
                with open(latest, "r", encoding="utf-8") as f:
                    report = json.load(f)
                
                dod_report = report.get("dod_report", {})
                
                STATE["last_dq"] = {
                    "ts": report.get("timestamp"),
                    "overall_passed": dod_report.get("overall_passed"),
                    "summary": dod_report.get("summary"),
                    "dod_results": dod_report.get("dod_results"),
                }
                
                passed = bool(STATE["last_dq"]["overall_passed"])
                if passed:
                    STATE["dq_fail_streak"] = 0
                else:
                    STATE["dq_fail_streak"] += 1
                
                log(f"[dq] Done (rc={rc}), passed={passed}, fail_streak={STATE['dq_fail_streak']}")
            else:
                log("[dq] No DQ reports found")
                
        except Exception as e:
            log(f"[dq] Exception: {e}")
        
        # Sleep until next interval
        elapsed = time.time() - t0
        sleep_time = max(5, VALIDATE_EVERY_MIN * 60 - elapsed)
        time.sleep(sleep_time)

class Handler(BaseHTTPRequestHandler):
    """HTTP handler for minimal UI"""
    
    def log_message(self, format, *args):
        """Suppress HTTP access logs"""
        pass
    
    def _txt(self, code, text):
        """Send plain text response"""
        body = text.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    
    def do_GET(self):
        """Handle GET requests"""
        path = self.path.split("?")[0]  # Remove query string
        
        if path == "/":
            # Main status page with auto-refresh
            uptime = "N/A"
            if STATE["last_start"]:
                try:
                    start = datetime.fromisoformat(STATE["last_start"].replace("Z", "+00:00"))
                    now = datetime.utcnow().replace(tzinfo=start.tzinfo)
                    uptime = str(now - start).split(".")[0]
                except:
                    uptime = "N/A"
            
            dq_status = "OK"
            if STATE["last_dq"]:
                passed = STATE["last_dq"].get("overall_passed")
                if passed is False:
                    dq_status = f"FAIL (streak={STATE['dq_fail_streak']})"
                elif passed is True:
                    dq_status = "PASS"
            
            lines = [
                "=" * 80,
                "harvestd - OFI+CVD 数据采集守护进程",
                "=" * 80,
                f"时间 (UTC): {datetime.utcnow().isoformat()}Z",
                f"状态: {'运行中' if STATE['running'] else '已停止'}",
                f"运行时长: {uptime}",
                "",
                "采集进程:",
                f"  PID: {STATE['harvester_pid'] or 'N/A'}",
                f"  重启次数: {STATE['restarts']}",
                f"  最后启动: {STATE['last_start']}",
                f"  最后退出: {STATE['last_exit']}",
                f"  最后心跳: {STATE['last_heartbeat']}",
                "",
                "数据质量:",
                f"  状态: {dq_status}",
                f"  最后检查: {STATE['last_dq']['ts'] if STATE['last_dq'] else 'N/A'}",
                f"  失败次数: {STATE['dq_fail_streak']}",
                "",
                "订单簿数据收集:",
                f"  启用状态: {'是' if STATE['orderbook_enabled'] else '否'}",
                f"  文件数量: {STATE['orderbook_files_count']}",
                f"  数据大小: {STATE['orderbook_data_size']:,} 字节",
                f"  最后更新: {STATE['orderbook_last_update'] or 'N/A'}",
                "",
                "最近日志:",
                "─" * 80,
                tail_log(50),
                "",
                "端点:",
                "  /         - 本状态页面",
                "  /health   - 健康检查 (200=正常, 503=降级)",
                "  /logs     - 最近日志",
                "  /dq       - 最新数据质量报告 (JSON)",
                "  /orderbook - 订单簿数据收集状态 (JSON)",
                "  /metrics  - Prometheus 指标",
                "=" * 80,
            ]
            
            txt = "\n".join(lines)
            
            # Wrap in HTML with auto-refresh
            html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="5">
<title>harvestd 状态</title>
<style>
body {{ font-family: 'Consolas', 'Courier New', monospace; font-size: 14px; background: #1e1e1e; color: #d4d4d4; margin: 20px; line-height: 1.4; }}
pre {{ margin: 0; white-space: pre-wrap; font-size: 14px; }}
</style>
</head>
<body>
<pre>{txt}</pre>
</body>
</html>"""
            
            html_bytes = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html_bytes)))
            self.end_headers()
            self.wfile.write(html_bytes)
            self.wfile.flush()
            
        elif path == "/health":
            # Health check
            pid_ok = STATE["harvester_pid"] is not None
            dq_ok = STATE["dq_fail_streak"] < DQ_FAIL_MAX_TOL
            health_ok = pid_ok and dq_ok
            
            status = "ok" if health_ok else "degraded"
            self._txt(200 if health_ok else 503, status)
            
        elif path == "/logs":
            # Recent logs only
            self._txt(200, tail_log(LOG_TAIL_LINES))
            
        elif path == "/dq":
            # Latest DQ report
            if STATE["last_dq"]:
                self._txt(200, json.dumps(STATE["last_dq"], ensure_ascii=False, indent=2))
            else:
                self._txt(404, "No DQ report available yet")
                
        elif path == "/orderbook":
            # 订单簿数据收集状态
            check_orderbook_status()  # 更新状态
            orderbook_status = {
                "enabled": STATE["orderbook_enabled"],
                "files_count": STATE["orderbook_files_count"],
                "data_size_bytes": STATE["orderbook_data_size"],
                "data_size_mb": round(STATE["orderbook_data_size"] / (1024 * 1024), 2),
                "last_update": STATE["orderbook_last_update"],
                "status": "active" if STATE["orderbook_files_count"] > 0 else "inactive"
            }
            self._txt(200, json.dumps(orderbook_status, ensure_ascii=False, indent=2))
                
        elif path == "/metrics":
            # Prometheus metrics
            lines = [
                "# HELP harvestd_restarts_total Total number of harvester restarts",
                "# TYPE harvestd_restarts_total counter",
                f"harvestd_restarts_total {STATE['restarts']}",
                "",
                "# HELP harvestd_dq_fail_streak Consecutive DQ failures",
                "# TYPE harvestd_dq_fail_streak gauge",
                f"harvestd_dq_fail_streak {STATE['dq_fail_streak']}",
                "",
                "# HELP harvestd_uptime_seconds Uptime in seconds",
                "# TYPE harvestd_uptime_seconds gauge",
            ]
            
            if STATE["last_start"]:
                try:
                    start = datetime.fromisoformat(STATE["last_start"].replace("Z", "+00:00"))
                    now = datetime.utcnow().replace(tzinfo=start.tzinfo)
                    uptime = (now - start).total_seconds()
                    lines.append(f"harvestd_uptime_seconds {uptime:.0f}")
                except:
                    lines.append("harvestd_uptime_seconds 0")
            else:
                lines.append("harvestd_uptime_seconds 0")
            
            self._txt(200, "\n".join(lines))
            
        else:
            self._txt(404, "not found")

def run_http():
    """Start HTTP server"""
    httpd = HTTPServer(("0.0.0.0", PORT), Handler)
    log(f"[ui] HTTP listening on :{PORT}")
    log(f"[ui] Access at http://localhost:{PORT}/")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        log("[ui] Stopping HTTP server")
    finally:
        httpd.server_close()

def main():
    """Main entry point"""
    # Setup signal handlers
    def signal_handler(sig, frame):
        log(f"[harvestd] Received signal {sig}, shutting down...")
        STATE["running"] = False
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    log("[harvestd] Starting daemon...")
    log(f"[harvestd] Config: PORT={PORT}, VALIDATE_EVERY_MIN={VALIDATE_EVERY_MIN}, RESTART_BACKOFF_MAX={RESTART_BACKOFF_MAX}")
    
    # Start worker threads
    harvester_thread = threading.Thread(target=run_harvester, daemon=True)
    validator_thread = threading.Thread(target=run_validator, daemon=True)
    
    harvester_thread.start()
    log("[harvestd] Harvester thread started")
    
    validator_thread.start()
    log("[harvestd] Validator thread started")
    
    # Run HTTP server in main thread
    try:
        run_http()
    finally:
        log("[harvestd] Shutting down...")

if __name__ == "__main__":
    main()
