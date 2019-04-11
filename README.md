# adPython

adPython is an [areaDetector][ad] plugin that embeds Python to allow
areaDetector image data to be processed by Python scripts.

The Python scripts are loaded from files thus allowing changes to be
made at run-time without recompiling.  Also, third party Python modules
may be used thus providing access to a large collection of modules such
as NumPy and OpenCV for image data processing.

## Dependencies

* [Python][python] 2.7.
* [NumPy][numpy].
* (Optional) [OpenCV][opencv] Python module.  (Used by some scripts.)

### Installing Python Module Dependencies with pip

```
pip install numpy
pip install opencv-python-headless
```

## Documentation

* [By Version][adpython]

[ad]: https://github.com/areaDetector/areaDetector
[adpython]: http://controls.diamond.ac.uk/downloads/support/adPython/
[numpy]: https://www.numpy.org/
[opencv]: https://opencv.org/
[python]: https://www.python.org/
