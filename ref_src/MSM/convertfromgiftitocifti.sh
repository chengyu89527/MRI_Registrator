#!/bin/bash
#   Emma Robinson, FMRIB Image Analysis Group
#   Copyright (C) 2012 University of Oxford
#
#   Part of FSL - FMRIB's Software Library
#   http://www.fmrib.ox.ac.uk/fsl
#   fsl@fmrib.ox.ac.uk
#
#   Developed at FMRIB (Oxford Centre for Functional Magnetic Resonance
#   Imaging of the Brain), Department of Clinical Neurology, Oxford
#   University, Oxford, UK
#
#
#   LICENCE
#
#   FMRIB Software Library, Release 6.0 (c) 2018, The University of
#   Oxford (the "Software")
#
#   The Software remains the property of the Oxford University Innovation
#   ("the University").
#
#   The Software is distributed "AS IS" under this Licence solely for
#   non-commercial use in the hope that it will be useful, but in order
#   that the University as a charitable foundation protects its assets for
#   the benefit of its educational and research purposes, the University
#   makes clear that no condition is made or to be implied, nor is any
#   warranty given or to be implied, as to the accuracy of the Software,
#   or that it will be suitable for any particular purpose or for use
#   under any specific conditions. Furthermore, the University disclaims
#   all responsibility for the use which is made of the Software. It
#   further disclaims any liability for the outcomes arising from using
#   the Software.
#
#   The Licensee agrees to indemnify the University and hold the
#   University harmless from and against any and all claims, damages and
#   liabilities asserted by third parties (including claims for
#   negligence) which arise directly or indirectly from the use of the
#   Software or the sale of any products based on the Software.
#
#   No part of the Software may be reproduced, modified, transmitted or
#   transferred in any form or by any means, electronic or mechanical,
#   without the express permission of the University. The permission of
#   the University is not required if the said reproduction, modification,
#   transmission or transference is done without financial return, the
#   conditions of this Licence are imposed upon the receiver of the
#   product, and all original and amended source code is included in any
#   transmitted product. You may be held legally responsible for any
#   copyright infringement that is caused or encouraged by your failure to
#   abide by these terms and conditions.
#
#   You are not permitted under this Licence to use this Software
#   commercially. Use for which any financial return is received shall be
#   defined as commercial use, and includes (1) integration of all or part
#   of the source code or the Software into a product for sale or license
#   by or on behalf of Licensee to third parties or (2) use of the
#   Software or any derivative of it for research with the final aim of
#   developing software products for sale or license to a third party or
#   (3) use of the Software or any derivative of it for research with the
#   final aim of developing non-software products for sale or license to a
#   third party, or (4) use of the Software to provide any service to an
#   external organisation for which payment is received. If you are
#   interested in using the Software commercially, please contact Oxford
#   University Innovation ("OUI"), the technology transfer company of the
#   University, to negotiate a licence. Contact details are:
#   fsl@innovation.ox.ac.uk quoting Reference Project 9564, FSL.
export LC_ALL=C

Usage() {
  echo "convertfromgiftitocifti <Subject> <dirpath>  <FeatType> <OutputRegName> <palettemax> <palettemin> "
  exit 1 
}

[ "$6" = "" ] && Usage

Subject=$1
dirname=$2
FeatType=$3
OutputRegName=$4
palettemax=$5
palettemin=$6

 wb_command -cifti-create-dense-timeseries ${dirname}/${Subject}.${FeatType}.${OutputRegName}.dtseries.nii -left-metric ${dirname}/${Subject}.L.${FeatType}.${OutputRegName}.func.gii -right-metric ${dirname}/${Subject}.R.${FeatType}.${OutputRegName}.func.gii
 
 wb_command -cifti-convert-to-scalar ${dirname}/${Subject}.${FeatType}.${OutputRegName}.dtseries.nii ROW ${dirname}/${Subject}.${FeatType}.${OutputRegName}.dscalar.nii 
 wb_command -set-map-name  ${dirname}/${Subject}.${FeatType}.${OutputRegName}.dscalar.nii 1 /${Subject}.${FeatType}.${OutputRegName}

 wb_command -cifti-palette  ${dirname}/${Subject}.${FeatType}.${OutputRegName}.dscalar.nii  MODE_USER_SCALE  ${dirname}/${Subject}.${FeatType}.${OutputRegName}.dscalar.nii -pos-user 0 $palettemax -neg-user 0 $palettemin -interpolate true -palette-name ROY-BIG-BL -disp-pos true -disp-neg true -disp-zero false
