#!/usr/bin/env dls-python

"""
Plugin to convert an NDAttribute to an NDArray

"""

from adPythonPlugin import AdPythonPlugin
import numpy
import logging


class AttributeToArray(AdPythonPlugin):

    def __init__(self):
        # Set logging level
        self.log.setLevel(logging.INFO)

        # Attributes
        self.attributes_missed_on_last_array = 0

        # Parameters
        params = dict(
            Status="Good",
            MissedAttributeCounter=0,
            ResetMissed=0,
            NumAttributes=0,)

        # Create attribute parameters
        for i in range(1, 9):
            params["Attribute{0}Name".format(i)] = ""
            params["Attribute{0}Value".format(i)] = 0.0

        AdPythonPlugin.__init__(self, params)

    def paramChanged(self):
        # Called every time a parameter changes

        # Reset missed attribute counter
        if self["ResetMissed"] == 1:
            self["MissedAttributeCounter"] = 0
            self["ResetMissed"] = 0

        # Constrain number of attributes between 0-8
        elif self["NumAttributes"] < 0:
            self["NumAttributes"] = 0
        elif self["NumAttributes"] > 8:
            self["NumAttributes"] = 8

    def processArray(self, arr, attr={}):  
        # Called every time plugin receives new array

        if self["NumAttributes"] > 0:
            attribute_array = self.getAttributesArray(attr)

            if self.attributes_missed_on_last_array > 0:
                self["Status"] = "Missed {0} attribute(s)".format(
                    self.attributes_missed_on_last_array)
                self["MissedAttributeCounter"] += self.attributes_missed_on_last_array
            else:
                self["Status"] = "Good"

            return attribute_array
        else:
            return

    def getAttributesArray(self, attr):
        # Build numpy array of attribute values
        num_attributes = self["NumAttributes"]
        attribute_array = numpy.zeros([num_attributes])
        self.attributes_missed_on_last_array = 0

        for i in range(num_attributes):
            attribute_value = self.getAttributeValue(
                attr, self["Attribute{0}Name".format(i+1)])
            self["Attribute{0}Value".format(i+1)] = attribute_value
            attribute_array[i] = attribute_value

        return attribute_array

    def getAttributeValue(self, attr, attribute):
        # Get a value of a single attribute
        try:
            attribute_value = float(attr[attribute])
            return attribute_value
        except:
            self.attributes_missed_on_last_array += 1
            self.log.error("No {0} attribute".format(attribute))
            return 0.0


if __name__ == "__main__":
    AttributeToArray().runOffline()