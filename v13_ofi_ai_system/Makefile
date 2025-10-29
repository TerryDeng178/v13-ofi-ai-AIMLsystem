# OFI_CVD_Fusion 测试 Makefile
# 
# 使用方法：
#   make test-unit       - 运行单元测试
#   make eval-synth      - 运行合成数据评测
#   make eval-offline    - 运行离线数据评测（若数据存在）
#   make report          - 汇总生成报告
#   make all             - 运行所有测试和评测

.PHONY: test-unit eval-synth eval-offline report all clean

# 变量定义
PYTHON = python
RESULTS_DIR = results

# 单元测试
test-unit:
	@echo "========================================"
	@echo "运行单元测试"
	@echo "========================================"
	$(PYTHON) tests/test_fusion_unit.py

# 合成数据评测
eval-synth:
	@echo "========================================"
	@echo "运行合成数据评测"
	@echo "========================================"
	$(PYTHON) scripts/run_fusion_synthetic_eval.py

# 离线数据评测
eval-offline:
	@echo "========================================"
	@echo "运行离线数据评测"
	@echo "========================================"
	$(PYTHON) scripts/run_fusion_offline_eval.py

# 生成报告（汇总现有结果）
report:
	@echo "========================================"
	@echo "生成测试报告"
	@echo "========================================"
	@if [ ! -f $(RESULTS_DIR)/fusion_test_report.md ]; then \
		echo "# OFI_CVD_Fusion 测试报告" > $(RESULTS_DIR)/fusion_test_report.md; \
		echo "" >> $(RESULTS_DIR)/fusion_test_report.md; \
		echo "本报告包含融合组件的测试结果与指标分析。" >> $(RESULTS_DIR)/fusion_test_report.md; \
		echo "" >> $(RESULTS_DIR)/fusion_test_report.md; \
	fi
	@echo "报告已生成: $(RESULTS_DIR)/fusion_test_report.md"
	@echo "指标汇总: $(RESULTS_DIR)/fusion_metrics_summary.csv"

# 运行所有测试和评测
all: test-unit eval-synth eval-offline report

# 清理结果文件
clean:
	@echo "清理结果文件..."
	rm -rf $(RESULTS_DIR)/fusion_metrics_summary.csv
	rm -rf $(RESULTS_DIR)/fusion_test_report.md
	@echo "清理完成"

