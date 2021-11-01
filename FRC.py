#!/usr/bin/env python
# This code was based off of: Copyright (c) 2018, Sami Koho, Molecular
# Microscopy & Spectroscopy, Italian Institute of Technology. All rights
# reserved. Please refer to the bottom of this code to read the license
# information.


import os
import numpy as np
import pandas
import sys

import miplib.ui.plots.image as showim
from miplib.psf import psfgen
from miplib.processing.deconvolution import deconvolve
from miplib.data.messages import image_writer_wrappers as imwrap
import miplib.data.io.read as imread
import miplib.processing.image as imops
from miplib.data.containers.image import Image

import miplib.analysis.resolution.fourier_ring_correlation as frc
from miplib.data.containers.fourier_correlation_data import FourierCorrelationDataCollection

import miplib.ui.plots.frc as frcplots

import miplib.ui.cli.miplib_entry_point_options as options
import urllib.request as dl

#In [2]
n_iterations = 50
args_list = ("image psf"
             " --max-nof-iterations={}  --first-estimate=image "
             " --blocks=1 --pad=0 --resolution-threshold-criterion=fixed "
             " --tv-lambda=0 --bin-delta=1  --frc-curve-fit-type=smooth-spline").format(n_iterations).split()

args = options.get_deconvolve_script_options(args_list)


# Image
data_dir = os.getcwd()
filename = input("Please enter the path to a tif file: ")
try:
    scriptDir = os.path.dirname(__file__)
    filename = os.path.join(scriptDir, filename)
except:
    print("Error:", filename, "is not a valid path")
    sys.exit(1)
full_path = os.path.join(data_dir, filename)

# Automatically dowload the file from figshare, if necessary.
if not os.path.exists(full_path):
        dl.urlretrieve("https://ndownloader.figshare.com/files/15202565", full_path)

image = imread.get_image(full_path, channel=0)

spacing = image.spacing
print ("The image dimensions are {} and spacing {} um.".format(image.shape, image.spacing))

image = Image(image - image.min(), image.spacing)
image_copy = Image(image.copy(), image.spacing)


frc_results = FourierCorrelationDataCollection()

frc_results[0] = frc.calculate_single_image_frc(image, args)

plotter = frcplots.FourierDataPlotter(frc_results)

plotter.plot_one(0)


fwhm = [frc_results[0].resolution['resolution'], ] * 2

psf_generator = psfgen.PsfFromFwhm(fwhm)

psf = psf_generator.xy()

showim.display_2d_images(imops.enhance_contrast(image_copy, percent_saturated=0.3),
                         psf,
                         image1_title="Original",
                         image2_title="PSF from FRC"

                         )


temp_dir = os.path.join(data_dir, "Temp")
if not os.path.exists(temp_dir):
    os.mkdir(temp_dir)

writer = imwrap.TiffImageWriter(data_dir)

task = deconvolve.DeconvolutionRL(image, psf, writer, args)
task.execute()


rl_result = task.get_result()

showim.display_2d_images(imops.enhance_contrast(image, percent_saturated=0.3),
                         imops.enhance_contrast(
    rl_result, percent_saturated=0.3),
    image1_title="Original",
    image2_title="Deconvolved, after {} iterations.".format(n_iterations))


frc_results[1] = frc.calculate_single_image_frc(rl_result, args)

plotter = frcplots.FourierDataPlotter(frc_results)
plotter.plot_all(custom_titles=("Original", "Deconvolved"))

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:

# * Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above
# copyright notice, this list of conditions and the following
# disclaimer in the documentation and / or other materials provided
# with the distribution.

# * Neither the name of the Molecular Microscopy and Spectroscopy
# research line, nor the names of its contributors may be used to
# endorse or promote products derived from this software without
# specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY COPYRIGHT HOLDER AND CONTRIBUTORS ''AS
# IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
# TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL COPYRIGHT HOLDER
# OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES(INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES
# LOSS OF USE, DATA, OR PROFITS OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# In addition to the terms of the license, we ask to acknowledge the use
# of the software in scientific articles by citing:

# Koho, S. et al. Fourier ring correlation simplifies image restoration in fluorescence
# microscopy. Nat. Commun. 10, 3103 (2019).

# Parts of the MIPLIB source code are based on previous BSD licensed
# open source projects:

# pyimagequalityranking:
# Copyright(c) 2015, Sami Koho, Laboratory of Biophysics, University of Turku.
# All rights reserved.

# supertomo:
# Copyright(c) 2014, Sami Koho, Laboratory of Biophysics, University of Turku.
# All rights reserved.

# iocbio-microscope:
# Copyright(c) 2009-2010, Laboratory of Systems Biology, Institute of
# Cybernetics at Tallinn University of Technology. All rights reserved
