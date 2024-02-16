import threading

import queue
import weakref


class BackgroundGenerator:
    def __init__(self, generator, max_prefetch=1):
        self.queue = queue.Queue(max_prefetch)
        self.generator = generator
        self.empty = False
        
        # Worker thread keeps a weak reference to this object, allowing this to be garbage collected while the thread is running
        self.worker_thread = threading.Thread(target=BackgroundGenerator.run, args=(weakref.proxy(self),), daemon=True)
        self.worker_thread.start()

    def run(self):
        try:
            for item in self.generator:
                while True:
                    try:
                        self.queue.put(item, timeout=1)
                        break
                    except queue.Full:
                        pass

            self.empty = True
        except ReferenceError:
            pass

    def next(self):
        if not self.worker_thread.is_alive():
            raise RuntimeError('Background worker thread died')
        if not self.queue.empty():
            return self.queue.get()
        
        if self.empty:
            raise StopIteration
            
        while True:
            if not self.worker_thread.is_alive():
                raise RuntimeError('Background worker thread died')
            try:
                return self.queue.get(timeout=1)
            except queue.Empty:
                pass
            
    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class BackgroundGeneratorPool:
    def __init__(self, generators, max_prefetch=1):
        self.queue = queue.Queue(max_prefetch * len(generators))
        self.daemon = True
        self.empty = [False for x in generators]
        self.generators = generators
        
        self.worker_threads = []
        for i,g in enumerate(generators):
            self.worker_threads.append(threading.Thread(target=BackgroundGeneratorPool.run, args=(weakref.proxy(self),i), daemon=True))
        for x in self.worker_threads:
            x.start()

    def run(self, index):
        try:
            for item in self.generators[index]:
                while True:
                    try:
                        self.queue.put(item, timeout=1)
                        break
                    except queue.Full:
                        pass

            self.empty[index] = True
        except ReferenceError:
            pass
        
    def next(self):
        # TODO: This can trigger if generators are not infinite
        if not any(x.is_alive() for x in self.worker_threads):
            raise RuntimeError('All background worker threads died')
            
        if not self.queue.empty():
            return self.queue.get()
        
        if all(self.empty):
            raise StopIteration
        
        while True:
            if not any(x.is_alive() for x in self.worker_threads):
                raise RuntimeError('All background worker threads died')
            try:
                return self.queue.get(timeout=1)
            except queue.Empty:
                pass
    
    def __next__(self):
        return self.next()

    def __iter__(self):
        return self
    
    
#decorator
class background:
    def __init__(self, max_prefetch=1):
        self.max_prefetch = max_prefetch
    def __call__(self, gen):
        def bg_generator(*args,**kwargs):
            return BackgroundGenerator(gen(*args,**kwargs), max_prefetch=self.max_prefetch)
        return bg_generator
