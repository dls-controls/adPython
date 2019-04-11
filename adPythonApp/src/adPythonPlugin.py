# our base class requires numpy, so make sure it's on the path here
# this step is only needed if numpy is an egg installed multi-version
try:
    from pkg_resources import require
    require("numpy")
except:
    pass

import imp, logging, numpy, multiprocessing, time, os, signal
from Queue import Empty

logging.basicConfig(format='%(asctime)s %(levelname)8s %(name)8s %(filename)s:%(lineno)d: %(message)s', level=logging.INFO)


# define a helper function that imports a python filename and returns an 
# instance of classname which is contained in it
def makePyInst(portname, filename, classname):
    # Create a logger associated with this portname
    log = logging.getLogger(portname)
    log.setLevel(logging.INFO) 
    log.info("Creating %s:%s with portname %s", 
        os.path.basename(filename), classname, portname)
    try:
        # This dance is needed to load a file explicitly from a filename
        f = open(filename)
        pymodule, ext = os.path.splitext(os.path.basename(filename))
        AdPythonPlugin.log = log        
        mod = imp.load_module(pymodule, f, filename, (ext, 'U', 1))
        f.close()
        # Get classname ref from this module and make an instance of it
        inst = getattr(mod, classname)()
        # Call paramChanged it might do some useful setup
        inst.paramChanged()
        return inst
    except:
        # Log the exception in the logger as the C caller will throw away the
        # exception text
        log.exception("Creating %s:%s threw exception", filename, classname)
        raise


def processArrayFromQueue(plugin):
    plugin.resultQueue.put((os.getpid(), "Worker started"))
    while True:
        if plugin.inputQueue.empty():
            time.sleep(0.001)
        else:
            threw = False
            (arr, attr, updated_params) = plugin.inputQueue.get()
            if not isinstance(arr, numpy.ndarray) and arr == "exit":
                plugin.resultQueue.close()
                plugin.arrayProcessExited.set()
                return
            for k, v in updated_params.items():
                plugin._params[k] = v
            try:
                new_array = plugin.processArray(arr, attr)
            except Exception as e:
                threw = True
                plugin.resultQueue.put(["failed:%s" % e, None, None])
            if not threw:
                try:
                    plugin.resultQueue.put((new_array, attr, plugin._params))
                except AssertionError:
                    # result queue was closed by the main thread, exit
                    return


class AdPythonPlugin(object):   
    # Will be our param dict
    _params = None
    # Will be our logger when used in conjunction with makePyInst()
    log = logging.getLogger("Offline")
    
    # init our param dict
    def __init__(self, params={}):

        self._params = dict(params)
        # self.log is the logger associated with AdPythonPlugin, copy it
        # and define it as the logger just for this instance...
        self.log = self.log
        self._timeout = None
        self._worker = 0
        self.inputQueue = None
        self.resultQueue = None
        self.processArrayProcess = None
        self.arrayProcessExited = None    # semaphore used bt worker to indicate it has exited
        self.arrayProcessRunning = multiprocessing.Event()
        self.notAwaitingResult = multiprocessing.Event()
        self.notAwaitingResult.set()
        self.initArrayProcess()

    # get a param value
    def __getitem__(self, param):
        return self._params[param]

    # set a param value 
    def __setitem__(self, param, value):
        assert param in self, "Param %s not in param lib" % param
        self._params[param] = value
 
    # see if param is supported
    def __contains__(self, param):
        return param in self._params
 
    # length of param dict
    def __len__(self):
        return len(self._params)

    # for if we want to print the dict 
    def __repr__(self):
        return repr(self._params)

    # iter
    def __iter__(self):
        return iter(self._params)

    def initArrayProcess(self):
        self.arrayProcessExited = multiprocessing.Event()
        oldInput = self.inputQueue
        oldResults = self.resultQueue
        self.inputQueue = multiprocessing.Queue()
        self.resultQueue = multiprocessing.Queue()
        self.processArrayProcess = multiprocessing.Process(target=processArrayFromQueue, args=(self,))
        self.processArrayProcess.daemon = True
        self.log.debug("spawned worker: %s" % self.processArrayProcess)
        self.processArrayProcess.start()
        (workerId, statusMessage) = self.resultQueue.get()
        self._worker = workerId
        self.log.info("new worker pid is %s" % workerId)
        self.arrayProcessRunning.set()

    def endArrayProcess(self):
        self.arrayProcessRunning.clear()
        self.inputQueue.put(["exit", None, None])
        self.resultQueue.put(["aborted:%s" % self._worker, None, None])
        self.notAwaitingResult.wait()
        didExit = self.arrayProcessExited.wait(timeout=0.01)
        self.inputQueue.close()
        try:
            self.processArrayProcess.terminate()
            os.kill(self._worker, signal.SIGKILL)
        except OSError:
            # process might have already gone
            pass
        if didExit:
            self.log.info("worker %s exited" % self._worker)
        else:
            self.log.info("worker %s killed" % self._worker)
            self.resultQueue.close()

    def abortProcessing(self):
        try:
            self.endArrayProcess()
        except Exception as e:
            self.log.exception(e)
        self.initArrayProcess()

    # called when parameter list changes
    def _paramChanged(self):
        try:
            self.log.debug("Param changed: %s", self._params)        
            return self.paramChanged()
        except:
            # Log the exception in the logger as the C caller will throw away 
            # the exception text        
            self.log.exception("Error calling paramChanged()")
            raise
    
    # default paramChanged does nothing
    def paramChanged(self):
        pass

    # Method called when array processing fails, default does nothing
    def processArrayFallback(self, arr, attr):
        return arr

    # called when a new array is generated
    def _processArray(self, arr, attr, timeout=None):
        if timeout == 0:
            timeout = None
        try:
            # Tell numpy that it does not own the data in arr, so it is read only
            # This should really be done at the C layer, but it's much easier here!        
            arr.flags.writeable = False
            # input dict of attributes is mutated instead of returned, hold on to ref so we
            # can update it when we get result back from the worker
            self._attr = attr
            if not self.arrayProcessRunning.wait(timeout=timeout):
                self.log.exception("Worker thread not running and timeout expired waiting for a new one")
                raise AssertionError("")
            self.notAwaitingResult.clear()
            self.inputQueue.put((arr, attr, self._params))
            return self.getResult(timeout)
        except Empty:
            # Worker didn't return processed array in time, call fallback method in main thread
            self.log.info("Timeout processing array; using fallback method")
            self.abortProcessing()
            return self.processArrayFallback(arr, attr)
        except AssertionError:
            # Queue was closed, abort was called in the C++ thread
            self.notAwaitingResult.set()
            self.log.info("Abort called whilst processing array; using fallback method")
            return self.processArrayFallback(arr, attr)
        except Exception:
            # Log the exception in the logger as the C caller will throw away
            # the exception text
            self.log.exception("Error calling processArray()")
            return None

    def hasResult(self):
        return not self.resultQueue.empty()

    def getResult(self, timeout=None):
        if timeout is not None:
            timeout /= 1000.0
        try:
            arr, attr, updated_params = self.resultQueue.get(timeout=timeout)
            if not isinstance(arr, numpy.ndarray):
                if arr.split(":")[0] == "aborted":
                    self.notAwaitingResult.set()
                    raise AssertionError("Abort was called on Worker")
                elif arr.split(":")[0] == "failed":
                    self.log.exception("Error getting array result from queue: %s" % arr.split(":")[1])
                    self.notAwaitingResult.set()
                    return None
            for k, v in attr.items():
                self._attr[k] = v  # input dict of attributes is mutated instead of returned
            for k, v in updated_params.items():
                self[k] = v
            self.notAwaitingResult.set()
            return arr
        except Empty:
            # Worker didn't return processed array in time, call fallback method in main thread
            self.notAwaitingResult.set()
            raise
        except AssertionError:
            # Queue was closed, abort was called in the C++ thread
            self.notAwaitingResult.set()
            raise
        except Exception as e:
            self.log.exception("Error getting array result from queue: %s" % e)
            self.notAwaitingResult.set()
            return None


    # called when run offline
    def runOffline(self, **ranges):
        from adPythonOffline import AdPythonOffline
        AdPythonOffline(self, **ranges)


if __name__=="__main__":
    # If run from the command line, assume we want the location of the numpy lib
    print numpy.get_include()
