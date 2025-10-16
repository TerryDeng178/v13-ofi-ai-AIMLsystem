#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V11币安真实数据训练流程
完整的下载数据 -> 训练模型 -> 回测评估流程
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from datetime import datetime
import json
import time

# 导入模块
from src.binance_data_downloader import BinanceDataDownloader
from src.v11_binance_backtest_trainer import V11BinanceBacktestTrainer

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class V11BinanceTrainingPipeline:
    """V11币安训练管道"""
    
    def __init__(self):
        self.downloader = BinanceDataDownloader()
        self.trainer = V11BinanceBacktestTrainer()
        
        # 训练配置
        self.config = {
            'symbol': 'ETHUSDT',
            'intervals': ['1m', '5m', '15m'],  # 重点训练1分钟数据
            'months': 6,
            'training_priorities': {
                '1m': 1,  # 最高优先级
                '5m': 2,  # 中等优先级
                '15m': 3   # 低优先级
            }
        }
        
        self.training_results = {}
        
        logger.info("V11币安训练管道初始化完成")
    
    def run_complete_training(self):
        """运行完整的训练流程"""
        logger.info("=" * 80)
        logger.info("V11币安完整训练流程")
        logger.info("=" * 80)
        
        start_time = datetime.now()
        
        try:
            # 阶段1: 数据下载
            logger.info("阶段1: 下载币安历史数据")
            self._download_data()
            
            # 阶段2: 数据验证
            logger.info("阶段2: 验证数据质量")
            self._validate_data()
            
            # 阶段3: 模型训练
            logger.info("阶段3: 训练V11模型")
            self._train_models()
            
            # 阶段4: 性能评估
            logger.info("阶段4: 评估模型性能")
            self._evaluate_performance()
            
            # 阶段5: 生成报告
            logger.info("阶段5: 生成训练报告")
            self._generate_report()
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            logger.info("=" * 80)
            logger.info("V11币安训练流程完成！")
            logger.info(f"总耗时: {duration}")
            logger.info("系统已准备好进行币安测试网实战！")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"训练流程失败: {e}")
            raise
    
    def _download_data(self):
        """下载数据"""
        logger.info("开始下载币安历史数据...")
        
        symbol = self.config['symbol']
        months = self.config['months']
        
        # 下载多时间间隔数据
        data_dict = self.downloader.download_multiple_intervals(symbol, months)
        
        if not data_dict:
            raise Exception("数据下载失败")
        
        # 保存数据摘要
        summary = self.downloader.save_data_summary(data_dict, symbol)
        
        self.training_results['data_summary'] = summary
        
        logger.info("数据下载完成！")
    
    def _validate_data(self):
        """验证数据质量"""
        logger.info("验证数据质量...")
        
        symbol = self.config['symbol']
        intervals = self.config['intervals']
        
        validation_results = {}
        
        for interval in intervals:
            try:
                # 加载数据
                df = self.trainer.load_binance_data(symbol, interval)
                
                # 验证数据质量
                quality_report = self.downloader.validate_data_quality(df)
                validation_results[interval] = quality_report
                
                logger.info(f"{interval} 数据验证:")
                logger.info(f"  记录数: {quality_report['total_records']:,}")
                logger.info(f"  价格波动率: {quality_report['price_range']['price_volatility']:.4f}")
                logger.info(f"  时间缺口: {quality_report['data_continuity']['time_gaps']}")
                
            except Exception as e:
                logger.error(f"验证 {interval} 数据失败: {e}")
                validation_results[interval] = {"error": str(e)}
        
        self.training_results['validation_results'] = validation_results
    
    def _train_models(self):
        """训练模型"""
        logger.info("开始训练V11模型...")
        
        symbol = self.config['symbol']
        intervals = self.config['intervals']
        priorities = self.config['training_priorities']
        
        # 按优先级排序
        sorted_intervals = sorted(intervals, key=lambda x: priorities.get(x, 999))
        
        training_results = {}
        
        for interval in sorted_intervals:
            logger.info(f"训练 {symbol} {interval} 模型...")
            
            try:
                # 训练模型
                result = self.trainer.train_and_evaluate(symbol, interval)
                training_results[interval] = result
                
                # 输出训练结果
                if 'error' not in result:
                    performance = result['performance']
                    logger.info(f"{interval} 训练完成:")
                    logger.info(f"  综合评分: {performance['overall_score']:.2f}")
                    logger.info(f"  年化收益: {performance['total_return']:.1%}")
                    logger.info(f"  夏普比率: {performance['sharpe_ratio']:.2f}")
                    logger.info(f"  交易准备: {'✅' if performance['trading_ready'] else '❌'}")
                else:
                    logger.error(f"{interval} 训练失败: {result['error']}")
                
                # 短暂休息
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"训练 {interval} 失败: {e}")
                training_results[interval] = {"error": str(e)}
        
        self.training_results['training_results'] = training_results
    
    def _evaluate_performance(self):
        """评估性能"""
        logger.info("评估整体性能...")
        
        training_results = self.training_results.get('training_results', {})
        
        # 找出最佳模型
        best_model = None
        best_score = -1
        
        for interval, result in training_results.items():
            if 'error' not in result and 'performance' in result:
                score = result['performance'].get('overall_score', 0)
                if score > best_score:
                    best_score = score
                    best_model = {
                        'interval': interval,
                        'result': result
                    }
        
        if best_model:
            logger.info(f"最佳模型: {best_model['interval']}")
            logger.info(f"最佳评分: {best_score:.2f}")
            
            # 检查是否达到交易标准
            performance = best_model['result']['performance']
            trading_ready = performance.get('trading_ready', False)
            
            if trading_ready:
                logger.info("🎉 系统已达到交易准备标准！")
            else:
                logger.info("⚠️ 系统尚未达到交易准备标准，需要进一步优化")
        else:
            logger.error("没有找到有效的训练结果")
        
        self.training_results['best_model'] = best_model
    
    def _generate_report(self):
        """生成训练报告"""
        logger.info("生成训练报告...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"v11_binance_complete_training_report_{timestamp}.json"
        
        # 添加报告元信息
        self.training_results['report_info'] = {
            'timestamp': timestamp,
            'symbol': self.config['symbol'],
            'intervals': self.config['intervals'],
            'months': self.config['months'],
            'pipeline_version': '1.0'
        }
        
        # 保存报告
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.training_results, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"训练报告已保存到: {report_file}")
        
        # 生成简要总结
        self._print_training_summary()
    
    def _print_training_summary(self):
        """打印训练总结"""
        logger.info("=" * 80)
        logger.info("V11币安训练总结")
        logger.info("=" * 80)
        
        # 数据摘要
        data_summary = self.training_results.get('data_summary', {})
        if data_summary:
            logger.info("数据摘要:")
            for interval, info in data_summary.get('intervals', {}).items():
                logger.info(f"  {interval}: {info.get('total_records', 0):,} 条记录")
        
        # 训练结果
        training_results = self.training_results.get('training_results', {})
        if training_results:
            logger.info("\n训练结果:")
            for interval, result in training_results.items():
                if 'error' not in result and 'performance' in result:
                    perf = result['performance']
                    logger.info(f"  {interval}: 评分={perf.get('overall_score', 0):.2f}, "
                              f"收益={perf.get('total_return', 0):.1%}, "
                              f"夏普={perf.get('sharpe_ratio', 0):.2f}")
                else:
                    logger.info(f"  {interval}: 训练失败")
        
        # 最佳模型
        best_model = self.training_results.get('best_model')
        if best_model:
            logger.info(f"\n最佳模型: {best_model['interval']}")
            perf = best_model['result']['performance']
            logger.info(f"  综合评分: {perf.get('overall_score', 0):.2f}")
            logger.info(f"  年化收益: {perf.get('total_return', 0):.1%}")
            logger.info(f"  夏普比率: {perf.get('sharpe_ratio', 0):.2f}")
            logger.info(f"  最大回撤: {perf.get('max_drawdown', 0):.1%}")
            logger.info(f"  交易准备: {'✅' if perf.get('trading_ready', False) else '❌'}")
        
        logger.info("=" * 80)


def main():
    """主函数"""
    logger.info("=" * 80)
    logger.info("V11币安真实数据训练系统")
    logger.info("=" * 80)
    
    # 创建训练管道
    pipeline = V11BinanceTrainingPipeline()
    
    # 运行完整训练流程
    pipeline.run_complete_training()


if __name__ == "__main__":
    main()
