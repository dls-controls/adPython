import copy

from iocbuilder import Device, AutoSubstitution
from iocbuilder.arginfo import *

from iocbuilder.modules.asyn import Asyn, AsynPort
from iocbuilder.modules.ADCore import ADCore, NDPluginBaseTemplate, includesTemplates, makeTemplateInstance

class AdPython(Device):
    '''Library dependencies for adPython'''
    Dependencies = (ADCore,)
    # Device attributes
    LibFileList = ['adPython']
    DbdFileList = ['adPythonPlugin']
    AutoInstantiate = True

@includesTemplates(NDPluginBaseTemplate)
class _adPythonBase(AutoSubstitution):
    '''This plugin Works out the area and tip of a sample'''
    TemplateFile = "adPythonPlugin.template"
    
class adPythonPlugin(AsynPort):
    """This plugin creates an adPython object"""
    # This tells xmlbuilder to use PORT instead of name as the row ID
    UniqueName = "PORT"

    _SpecificTemplate = _adPythonBase
    Dependencies = (AdPython,)

    def __init__(self, classname, PORT, NDARRAY_PORT, QUEUE = 5, BLOCK = 0, NDARRAY_ADDR = 0, BUFFERS = 50, MEMORY = 0,
                 CUSTOM_CLASS="", CUSTOM_FILE="", CUSTOM_NINT=0, CUSTOM_NDOUBLE=0, CUSTOM_NINTARR=0, CUSTOM_NDOUBLEARR=0, **args):
        # Init the superclass (AsynPort)
        self.__super.__init__(PORT)
        # Update the attributes of self from the commandline args
        self.__dict__.update(locals())
        # Make an instance of our template
        makeTemplateInstance(self._SpecificTemplate, locals(), args)
        # Arguments used for the class associated template/s
        _tmpargs = copy.deepcopy(args)
        _tmpargs['PORT'] = PORT
        # Init the python classname specific class
        if classname == "Custom":
            class _tmpint(AutoSubstitution):
                ModuleName = adPythonPlugin.ModuleName
                TemplateFile = "adPythonCustomInt.template"

            for index in range(1, CUSTOM_NINT+1):
                _tmpint(N=index, **_tmpargs)

            class _tmpdouble(AutoSubstitution):
                ModuleName = adPythonPlugin.ModuleName
                TemplateFile = "adPythonCustomDouble.template"

            for index in range(1, CUSTOM_NDOUBLE+1):
                _tmpdouble(N=index, **_tmpargs)
                
            class _tmpintarray(AutoSubstitution):
                ModuleName = adPythonPlugin.ModuleName
                TemplateFile = "adPythonCustomIntArray.template"

            for index in range(1, CUSTOM_NINTARR+1):
                _tmpintarray(N=index, **_tmpargs)

            class _tmpdoublearray(AutoSubstitution):
                ModuleName = adPythonPlugin.ModuleName
                TemplateFile = "adPythonCustomDoubleArray.template"

            for index in range(1, CUSTOM_NDOUBLEARR+1):
                _tmpdoublearray(N=index, **_tmpargs)

            self.filename = CUSTOM_FILE
            self.classname = CUSTOM_CLASS
        else:
            class _tmp(AutoSubstitution):
                ModuleName = adPythonPlugin.ModuleName
                TrueName = "_adPython%s" % classname
                TemplateFile = "adPython%s.template" % classname

            _tmp(**filter_dict(_tmpargs, _tmp.ArgInfo.Names()))

            self.filename = "$(ADPYTHON)/adPythonApp/scripts/adPython%s.py" % classname

        self.Configure = 'adPythonPluginConfigure'

    def Initialise(self):
        print '# %(Configure)s(portName, filename, classname, queueSize, '\
            'blockingCallbacks, NDArrayPort, NDArrayAddr, maxBuffers, ' \
            'maxMemory)' % self.__dict__
        print '%(Configure)s("%(PORT)s", "%(filename)s", "%(classname)s", %(QUEUE)d, ' \
            '%(BLOCK)d, "%(NDARRAY_PORT)s", %(NDARRAY_ADDR)s, %(BUFFERS)d, ' \
            '%(MEMORY)d)' % self.__dict__

    # __init__ arguments
    ArgInfo = _SpecificTemplate.ArgInfo + makeArgInfo(__init__,
        classname = Choice('Predefined python class to use', [
            "Morph", "Focus", "Template", "BarCode", "Transfer", "Mitegen",
            "Circle", "DataMatrix", "Gaussian2DFitter", "PowerMean",
            "MxSampleDetect","Rotate", "Custom"]),
        PORT = Simple('Port name for the plugin', str),
        QUEUE = Simple('Input array queue size', int),
        BLOCK = Simple('Blocking callbacks?', int),
        NDARRAY_PORT = Ident('Input array port', AsynPort),
        NDARRAY_ADDR = Simple('Input array port address', int),
        BUFFERS = Simple('Maximum number of NDArray buffers to be created for '
            'plugin callbacks', int),
        MEMORY = Simple('Max memory to allocate, should be maxw*maxh*nbuffer '
            'for driver and all attached plugins', int),

        CUSTOM_CLASS = Simple('Class name used when setting a custom class', str),
        CUSTOM_FILE = Simple('Python file path used when setting a custom class', str),
        CUSTOM_NINT = Simple('Number of integer parameters in the selected custom class (i.e: int1, int2 ...)', int),
        CUSTOM_NDOUBLE = Simple('Number of double parameters in the selected custom class (i.e: double1, double2 ...)',
                                int))


