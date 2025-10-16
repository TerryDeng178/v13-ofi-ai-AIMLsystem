import logging, logging.handlers, queue

class DropQueueHandler(logging.handlers.QueueHandler):
    def __init__(self, q):
        super().__init__(q); self.drops = 0; self.max_depth = 0
    def emit(self, record):
        try:
            d = self.queue.qsize()
            if d > self.max_depth: self.max_depth = d
            self.queue.put_nowait(record)
        except queue.Full:
            self.drops += 1

def _make_rotating_handler(path, rotate='interval', rotate_sec=60, max_bytes=5_000_000, backups=7):
    if rotate == 'size':
        return logging.handlers.RotatingFileHandler(path, maxBytes=max_bytes, backupCount=backups, encoding='utf-8')
    else:
        return logging.handlers.TimedRotatingFileHandler(path, when='s', interval=rotate_sec, backupCount=backups, encoding='utf-8')

def setup_async_logging(name: str, log_path: str, *, rotate='interval', rotate_sec=60, max_bytes=5_000_000,
                        backups=7, level=logging.INFO, queue_max=10000, to_console=True):
    q = queue.Queue(maxsize=queue_max)
    qh = DropQueueHandler(q)
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(name)s - %(message)s')
    fh = _make_rotating_handler(log_path, rotate, rotate_sec, max_bytes, backups); fh.setFormatter(fmt)
    handlers = [fh]
    if to_console:
        ch = logging.StreamHandler(); ch.setFormatter(fmt); handlers.append(ch)
    listener = logging.handlers.QueueListener(q, *handlers)
    logger = logging.getLogger(name); logger.setLevel(level); logger.handlers = []; logger.addHandler(qh); logger.propagate = False
    listener.start()
    return logger, listener, qh

def sample_queue_metrics(qh: DropQueueHandler):
    try: depth = qh.queue.qsize()
    except Exception: depth = 0
    return {'depth': depth, 'max_depth': getattr(qh, 'max_depth', 0), 'drops': getattr(qh, 'drops', 0)}
