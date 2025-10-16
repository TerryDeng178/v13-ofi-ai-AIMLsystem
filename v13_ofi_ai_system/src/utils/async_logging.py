"""
异步日志工具模块
仅使用标准库，实现非阻塞日志、轮转与保留
"""
import logging
import logging.handlers
import queue


class DropQueueHandler(logging.handlers.QueueHandler):
    """支持丢弃统计的队列日志处理器"""
    
    def __init__(self, q):
        super().__init__(q)
        self.drops = 0
        self.max_depth = 0
    
    def emit(self, record):
        """发送日志记录到队列，满时丢弃并统计"""
        try:
            d = self.queue.qsize()
            if d > self.max_depth:
                self.max_depth = d
            self.queue.put_nowait(record)
        except queue.Full:
            self.drops += 1


def _make_rotating_handler(path, rotate='interval', rotate_sec=60, max_bytes=5_000_000, backups=7):
    """创建轮转日志处理器
    
    Args:
        path: 日志文件路径
        rotate: 轮转模式 ('interval' 或 'size')
        rotate_sec: 时间轮转间隔（秒）
        max_bytes: 大小轮转阈值（字节）
        backups: 保留备份数量
    
    Returns:
        RotatingFileHandler 或 TimedRotatingFileHandler
    """
    if rotate == 'size':
        return logging.handlers.RotatingFileHandler(
            path, 
            maxBytes=max_bytes, 
            backupCount=backups, 
            encoding='utf-8'
        )
    else:  # interval
        return logging.handlers.TimedRotatingFileHandler(
            path, 
            when='S',  # 秒级轮转
            interval=rotate_sec, 
            backupCount=backups, 
            encoding='utf-8'
        )


def setup_async_logging(name: str, log_path: str, *, 
                       rotate='interval', 
                       rotate_sec=60, 
                       max_bytes=5_000_000,
                       backups=7, 
                       level=logging.INFO, 
                       queue_max=10000, 
                       to_console=True):
    """设置异步日志系统
    
    Args:
        name: logger名称
        log_path: 日志文件路径
        rotate: 轮转模式 ('interval' 或 'size')
        rotate_sec: 时间轮转间隔（秒）
        max_bytes: 大小轮转阈值（字节）
        backups: 保留备份数量
        level: 日志级别
        queue_max: 队列最大容量
        to_console: 是否同时输出到控制台
    
    Returns:
        (logger, listener, queue_handler) 元组
    """
    # 创建队列和队列处理器
    q = queue.Queue(maxsize=queue_max)
    qh = DropQueueHandler(q)
    
    # 创建格式化器
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(name)s - %(message)s')
    
    # 创建文件处理器（带轮转）
    fh = _make_rotating_handler(log_path, rotate, rotate_sec, max_bytes, backups)
    fh.setFormatter(fmt)
    
    # 处理器列表
    handlers = [fh]
    
    # 可选：添加控制台处理器
    if to_console:
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        handlers.append(ch)
    
    # 创建并启动监听器（在独立线程中处理日志）
    listener = logging.handlers.QueueListener(q, *handlers, respect_handler_level=True)
    
    # 配置logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []
    logger.addHandler(qh)
    logger.propagate = False
    
    # 启动监听器
    listener.start()
    
    return logger, listener, qh


def sample_queue_metrics(qh: DropQueueHandler):
    """采样队列指标
    
    Args:
        qh: DropQueueHandler实例
    
    Returns:
        包含depth, max_depth, drops的字典
    """
    try:
        depth = qh.queue.qsize()
    except Exception:
        depth = 0
    
    return {
        'depth': depth,
        'max_depth': getattr(qh, 'max_depth', 0),
        'drops': getattr(qh, 'drops', 0)
    }

