#!/usr/bin/env python3
"""
简化的Prometheus指标服务器
避免UTF-8编码问题
"""

import http.server
import socketserver
import random
import time

def generate_metrics():
    """生成策略模式相关的Prometheus指标"""
    
    # 模拟当前模式（0=quiet, 1=active）
    current_mode = random.choice([0, 1])
    
    # 模拟市场活动指标
    trades_per_min = random.uniform(100, 800)
    quotes_per_sec = random.uniform(20, 150)
    spread_bps = random.uniform(1, 10)
    volatility_bps = random.uniform(5, 50)
    volume_usd = random.uniform(500000, 5000000)
    
    # 模拟参数更新耗时
    update_duration = random.uniform(10, 200)
    
    # 模拟时间戳
    current_time = int(time.time() * 1000)
    
    metrics = f"""# HELP strategy_mode_active Current strategy mode (0=quiet, 1=active)
# TYPE strategy_mode_active gauge
strategy_mode_active{{env="testing",symbol="BTCUSDT"}} {current_mode}

# HELP strategy_mode_last_change_timestamp Timestamp of last mode change
# TYPE strategy_mode_last_change_timestamp gauge
strategy_mode_last_change_timestamp{{env="testing",symbol="BTCUSDT"}} {current_time - random.randint(1000, 3600000)}

# HELP strategy_mode_transitions_total Total number of mode transitions
# TYPE strategy_mode_transitions_total counter
strategy_mode_transitions_total{{env="testing",symbol="BTCUSDT",reason="schedule",from="quiet",to="active"}} {random.randint(5, 20)}
strategy_mode_transitions_total{{env="testing",symbol="BTCUSDT",reason="market",from="active",to="quiet"}} {random.randint(3, 15)}
strategy_mode_transitions_total{{env="testing",symbol="BTCUSDT",reason="manual",from="quiet",to="active"}} {random.randint(1, 5)}

# HELP strategy_time_in_mode_seconds_total Total time spent in each mode
# TYPE strategy_time_in_mode_seconds_total counter
strategy_time_in_mode_seconds_total{{env="testing",symbol="BTCUSDT",mode="active"}} {random.randint(3600, 7200)}
strategy_time_in_mode_seconds_total{{env="testing",symbol="BTCUSDT",mode="quiet"}} {random.randint(1800, 3600)}

# HELP strategy_trigger_trades_per_min Number of trades per minute
# TYPE strategy_trigger_trades_per_min gauge
strategy_trigger_trades_per_min{{env="testing",symbol="BTCUSDT"}} {trades_per_min:.2f}

# HELP strategy_trigger_quote_updates_per_sec Number of quote updates per second
# TYPE strategy_trigger_quote_updates_per_sec gauge
strategy_trigger_quote_updates_per_sec{{env="testing",symbol="BTCUSDT"}} {quotes_per_sec:.2f}

# HELP strategy_trigger_spread_bps Spread in basis points
# TYPE strategy_trigger_spread_bps gauge
strategy_trigger_spread_bps{{env="testing",symbol="BTCUSDT"}} {spread_bps:.2f}

# HELP strategy_trigger_volatility_bps Volatility in basis points
# TYPE strategy_trigger_volatility_bps gauge
strategy_trigger_volatility_bps{{env="testing",symbol="BTCUSDT"}} {volatility_bps:.2f}

# HELP strategy_trigger_volume_usd Volume in USD
# TYPE strategy_trigger_volume_usd gauge
strategy_trigger_volume_usd{{env="testing",symbol="BTCUSDT"}} {volume_usd:.2f}

# HELP strategy_params_update_duration_ms_bucket Parameter update duration histogram
# TYPE strategy_params_update_duration_ms_bucket histogram
strategy_params_update_duration_ms_bucket{{env="testing",le="10"}} {random.randint(0, 5)}
strategy_params_update_duration_ms_bucket{{env="testing",le="50"}} {random.randint(5, 20)}
strategy_params_update_duration_ms_bucket{{env="testing",le="100"}} {random.randint(15, 35)}
strategy_params_update_duration_ms_bucket{{env="testing",le="200"}} {random.randint(30, 45)}
strategy_params_update_duration_ms_bucket{{env="testing",le="+Inf"}} {random.randint(40, 50)}
strategy_params_update_duration_ms_sum{{env="testing"}} {update_duration * random.randint(40, 50):.2f}
strategy_params_update_duration_ms_count{{env="testing"}} {random.randint(40, 50)}

# HELP strategy_params_update_failures_total Total parameter update failures
# TYPE strategy_params_update_failures_total counter
strategy_params_update_failures_total{{env="testing",module="cvd"}} {random.randint(0, 2)}
strategy_params_update_failures_total{{env="testing",module="ofi"}} {random.randint(0, 1)}
strategy_params_update_failures_total{{env="testing",module="risk"}} {random.randint(0, 1)}

# HELP strategy_metrics_last_scrape_timestamp Timestamp of last metrics scrape
# TYPE strategy_metrics_last_scrape_timestamp gauge
strategy_metrics_last_scrape_timestamp{{env="testing"}} {current_time}
"""

    return metrics

class MetricsHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/metrics':
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(generate_metrics().encode('utf-8'))
        elif self.path == '/health':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"status": "healthy"}')
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # 减少日志输出
        pass

def start_metrics_server(port=8000):
    """启动指标服务器"""
    with socketserver.TCPServer(("", port), MetricsHandler) as httpd:
        print(f"指标服务器启动: http://localhost:{port}/metrics")
        print(f"健康检查: http://localhost:{port}/health")
        print("按 Ctrl+C 停止服务器")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("服务器已停止")

if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    start_metrics_server(port)
