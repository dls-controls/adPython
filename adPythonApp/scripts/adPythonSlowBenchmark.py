#!/usr/bin/env dls-python

from pkg_resources import require
require("fit_lib == 1.3")
require("scipy == 0.10.1")
require("cothread==2.15")

from adPythonPlugin import AdPythonPlugin
import cv2
import numpy
import scipy.ndimage


class SlowBenchmark(AdPythonPlugin):
    tempCounter = 0
    def __init__(self):
        # The default logging level is INFO.
        # Comment this line to set debug logging off
        # self.log.setLevel(logging.DEBUG) 
        # Make inputs and ouptuts list
        params = dict()
        AdPythonPlugin.__init__(self, params)

    def processArray(self, arr, attr={}):

        arr = numpy.float32(arr)
        # Run a median filter over the image to remove the spikes due to dead pixels.
        arr = numpy.float32(scipy.ndimage.median_filter(arr, size=3))
        attr["messedWith"] = True
        ret, thresh = cv2.threshold(arr, 127, 255, 0)
        thresh = numpy.uint8(thresh)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(arr, contours, -1, (255,255,255))
        # return the resultant array.
        return numpy.uint8(arr)

if __name__=="__main__":
    SlowBenchmark().runOffline()
