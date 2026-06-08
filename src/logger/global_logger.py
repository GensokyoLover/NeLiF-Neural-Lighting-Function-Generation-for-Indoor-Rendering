import logging

class GlobalLogger():
    '''
    A simple implementation of global logger (singleton).
    #! It's not threading-safe now !!！
    '''

    #TODO: threading-safe; multiprocessing-safe

    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = object.__new__(cls, *args, **kwargs)
            cls._instance._start(*args, **kwargs)
        return cls._instance

    def _start(self, *args, **kwargs):
        self.logger = logging.getLogger('main')
        self.logger.setLevel(kwargs.get('level', logging.WARNING))

        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)
    
