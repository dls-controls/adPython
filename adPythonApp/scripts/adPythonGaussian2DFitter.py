#!/usr/bin/env dls-python

from pkg_resources import require
require("fit_lib == 1.3")
require("scipy == 0.10.1")

from adPythonPlugin import AdPythonPlugin
import numpy
from fit_lib import fit_lib
from fit_lib_temp.levmar import FitError
from fit_lib_temp.fit_lib_temp import doFit2dGaussian, convert_abc
import scipy.ndimage

# copied from fit_lib for to debug


def centre_of_mass(image):
    m00 = numpy.sum(image)
    m10 = 0
    m01 = 0
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            m10 += x * image[x, y]
            m01 += y * image[x, y]
    cx = int(m10/m00)
    cy = int(m01/m00)
    sx = numpy.std(image[:, cy])
    sy = numpy.std(image[cx, :])
    return cx, cy, sx, sy, int(m00)

class Gaussian2DFitter(AdPythonPlugin):
    def __init__(self):
        # The default logging level is INFO.
        # Comment this line to set debug logging off
        # self.log.setLevel(logging.DEBUG) 
        # Make inputs and ouptuts list
        params = dict(PeakHeight=1,
                      OriginX=2,
                      OriginY=3,
                      Baseline=3.,
                      SigmaX=1.0,
                      SigmaY=2.0,
                      Angle=3.0,
                      Error=2.0,
                      FitWindowSize=3,
                      FitThinning=5,
                      Maxiter=20,
                      FitStatus="",
                      OverlayROI=0,
                      OverlayElipse=0,
                      OverlayCross=0,
                      OutputType=0,
                      FitType=-1
                      )
        AdPythonPlugin.__init__(self, params)

    # def paramChanged(self):
    #     # one of our input parameters has changed
    #     # just log it for now, do nothing.
    #     self.log.debug("Parameter has been changed %s", self)

    def processArray(self, arr, attr={}):
        # Called when the plugin gets a new array
        # arr is a numpy array
        # attr is an attribute dictionary that will be attached to the array
        self["FitType"] = -1
        self["FitStatus"] = "Processing array..."
        # Convert the array to a float so that we do not overflow during processing.
        arr2 = numpy.float_(arr)

        # Run a median filter over the image to remove the spikes due to dead pixels.
        arr2 = scipy.ndimage.median_filter(arr2, size=3)
        try:
            fit, error, results = doFit2dGaussian(
                arr2, thinning=(self["FitThinning"], self["FitThinning"]), #self.FitThinning
                window_size=self["FitWindowSize"], maxiter=self["Maxiter"], #self.FitWindowSize   self.maxiter
                ROI=None, gamma=(0, 255), ##[[150, 150],[100, 100]]
                extra_data=True)
            # fit outputs in terms of ABC we want sigma x, sigma y and angle.
            s_x, s_y, th = convert_abc(*fit[4:7])
            if any([fit[i+2] < -arr2.shape[i] or fit[i+2] > 2*arr2.shape[i] for i in [0, 1]]):
                raise FitError("Fit out of range")

        except FitError:
            self["FitStatus"] = "Fit error (using CoM as fallback)"
            self["FitType"] = 1
            cx, cy, s_x, s_y, h0 = centre_of_mass(arr2)
            th = 0.0
            fit = [0.0, h0, cx, cy]
            error = 0.0
        else:
            self["FitStatus"] = "Gaussian Fit OK"
            self["FitType"] = 0

        # Write out to the EDM output parameters.
        self["Baseline"] = float(fit[0])
        self["PeakHeight"] = int(fit[1])
        self["OriginX"] = int(fit[2])
        self["OriginY"] = int(fit[3])
        self["SigmaX"] = s_x
        self["SigmaY"] = s_y
        self["Angle"] = th
        self["Error"] = float(error)

        if self["OutputType"] == 1:
            # create the model output and take a difference to the original data.
            grid = fit_lib.flatten_grid(fit_lib.create_grid(arr.shape))
            arr = arr2 - fit_lib.Gaussian2d(fit, grid).reshape(arr.shape)
            arr = numpy.uint8(arr)

        # Add the annotations
        def plot_ab_axis(image, orig_x, orig_y, theta, ax_size=70, col=256):
            '''Creates image overlay for crosshairs.'''
            theta = -theta * numpy.pi / 180  # converting to radians
            # Create an array of zeros the same size as the original image.
            overlay_cross = numpy.zeros_like(image)

            x_len = overlay_cross.shape[0]
            y_len = overlay_cross.shape[1]

            # Draw cross pixel by pixel by setting each pixel to 256 (i.e. white)
            for axs in range(0, ax_size):
                ulimb = (int(orig_x + axs * numpy.cos(theta)), int(orig_y + axs * numpy.sin(theta)))
                llimb = (int(orig_x - axs * numpy.cos(theta)), int(orig_y - axs * numpy.sin(theta)))
                ulima = (int(orig_x + axs * numpy.sin(theta)), int(orig_y - axs * numpy.cos(theta)))
                llima = (int(orig_x - axs * numpy.sin(theta)), int(orig_y + axs * numpy.cos(theta)))


                if is_inside_image_boundary(ulimb[0], ulimb[1], x_len, y_len):
                    overlay_cross[ulimb] = col
                if is_inside_image_boundary(llimb[0], llimb[1], x_len, y_len):
                    overlay_cross[llimb] = col
                if is_inside_image_boundary(ulima[0], ulima[1], x_len, y_len):
                    overlay_cross[ulima] = col
                if is_inside_image_boundary(llima[0], llima[1], x_len, y_len):
                    overlay_cross[llima] = col

            return overlay_cross

        def plot_elipse(image, orig_x, orig_y, sig_x, sig_y, theta, col):
            '''Plots an elipse on the given axis of interest.'''
            # Create an array of zeros the same size as the original image.
            overlay_elipse = numpy.zeros_like(image)
            ex_vec = numpy.arange(-1, 1, 0.01) * sig_x
            ey_vec = numpy.sqrt(numpy.square(sig_y) * (1. - (numpy.square(ex_vec) / numpy.square(sig_x))))
            ex_vec = numpy.hstack([ex_vec, -ex_vec])
            ey_vec = numpy.hstack([ey_vec, -ey_vec])
            theta = theta * numpy.pi / 180  # converting to radians
            # converting to r, theta and adding additional theta term
            r = numpy.sqrt(ex_vec*ex_vec + ey_vec*ey_vec)
            t = numpy.arctan(ey_vec/ex_vec) - theta
            # Converting back to [x,y]
            x_len = len(ex_vec)
            x_seg = int(numpy.floor(x_len/2))
            ex_vec[:x_seg] = r[:x_seg] * numpy.cos(t[:x_seg])
            ey_vec[:x_seg] = r[:x_seg] * numpy.sin(t[:x_seg])
            ex_vec[x_seg:] = -r[x_seg:] * numpy.cos(t[x_seg:])
            ey_vec[x_seg:] = -r[x_seg:] * numpy.sin(t[x_seg:])
            # Moving the origin
            ex_vec = ex_vec + orig_x
            ey_vec = ey_vec + orig_y
            point_list = zip(ex_vec, ey_vec)

            for nf in point_list:
                if is_inside_image_boundary(int(nf[0]), int(nf[1]), overlay_elipse.shape[0], overlay_elipse.shape[1]):
                    overlay_elipse[int(nf[0]), int(nf[1])] = col
            return overlay_elipse

        def plot_ROI(image, results):
            '''Plots a box showing the region of interest used for the fit.'''
            # Create an array of zeros the same size as the original image.
            overlay_ROI = numpy.zeros_like(image)
            for ns in range(int(results.origin[1]), int(results.origin[1]) + results.extent[1]):
                overlay_ROI[(int(results.origin[0]), ns)] = 255
                overlay_ROI[(int(results.origin[0]) + results.extent[0]-1, ns)] = 255
            for nt in range(int(results.origin[0]), int(results.origin[0]) + results.extent[0]):
                overlay_ROI[(nt, int(results.origin[1]))] = 255
                overlay_ROI[(nt, int(results.origin[1]) + results.extent[1]-1)] = 255
            return overlay_ROI

        def apply_overlay(image, overlay):
            # Preferentially sets the pixel value to the overlay value if the overlay is not zero.
            out = numpy.where(overlay == 0, image, overlay)
            return out

        def is_inside_image_boundary(x, y, x_size, y_size):
            '''Checks if point (x,y) is within a x_size*y_size image'''
            if x < 0 or x >= x_size or y < 0 or y >= y_size:
                return False
            else:
                return True
        if fit[2] < arr.shape[0] and fit[3] < arr.shape[1]:
            cross_size = int(0.5*min([arr.shape[0] - fit[2], fit[2], arr.shape[1] - fit[3], fit[3], 40]))
        else:
            cross_size = 20
        if self["OverlayCross"] == 1:
            ol_cross = plot_ab_axis(arr, fit[2], fit[3], th, ax_size=cross_size, col=255)
            arr = apply_overlay(arr, ol_cross)


        ellipse_x = min((0.5*(arr.shape[0] - fit[2]), 0.5*fit[2], s_x))
        ellipse_y = min((0.5*(arr.shape[1] - fit[3]), 0.5*fit[3], s_y))
        if self["OverlayElipse"] == 1:
            ol_elipse = plot_elipse(arr, fit[2], fit[3], ellipse_x, ellipse_y, th, 255)
            arr = apply_overlay(arr, ol_elipse)

        if self["OverlayROI"] == 1:
            ol_ROI = plot_ROI(arr, results)
            arr = apply_overlay(arr, ol_ROI)

        # Write the attibute array which will be attached to the output array.
        #Note that we convert from the numpy
        # uint64 type to a python integer as we only handle python integers,
        # doubles and strings in the C code for now
        # Fitter results
        for param in self:
            attr[param] = self[param]
        # Write something to the logs
        self.log.debug("Array processed, baseline: %f, peak height: %d, origin x: %d, origin y: %d, sigma x: %f, sigma y: %f, angle: %f, error: %f, output: %d", self["Baseline"], self["PeakHeight"], self["OriginX"], self["OriginY"], self["SigmaX"],self["SigmaY"],self["Angle"], self["Error"], self["OutputType"])
        # return the resultant array.
        return arr

if __name__=="__main__":
    Gaussian2DFitter().runOffline(
        int1=256,            # This has range 0..255
        int2=500,        # This has range 0..255
        int3=500,        # This has range 0..255
        double1=(0,30,0.01), # This has range 0, 0.01, 0.02 ... 30
        double2=(0,30,0.01), # This has range 0, 0.01, 0.02 ... 30
        double3=(0,360,0.1)) # This has range 0, 0.1, 0.002 ... 360
    #     PeakHeight = 256,
    #     OriginX = (-500, 500,1), 
    #     OriginY = (-500,500,1),  
    #     Baseline = (-256, 256,1), 
    #     SigmaX = 30,
    #     SigmaY = 30, 
    #     Angle = (-360, 360, 1),
    #    Error = 500);
