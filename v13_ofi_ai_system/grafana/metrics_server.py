#!/usr/bin/env python3
"""
简单的Prometheus指标服务器
提供策略模式管理器的模拟指标
"""

import http.server
import socketserver
import threading
import time
import sys
import io

# 设置UTF-8编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from metrics_generator import generate_metrics

class MetricsHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/metrics':
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain; charset=utf-8')
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
        print(f"✅ 指标服务器启动: http://localhost:{port}/metrics")
        print(f"✅ 健康检查: http://localhost:{port}/health")
        print("按 Ctrl+C 停止服务器")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n🛑 服务器已停止")

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    start_metrics_server(port)
