/*  Multimodal Image Segmentation Tool (MIST)  */
/*  Eelke Visser  */

/*  Copyright (c) 2016 University of Oxford  */

/*  Part of FSL - FMRIB's Software Library
    http://www.fmrib.ox.ac.uk/fsl
    fsl@fmrib.ox.ac.uk

    Developed at FMRIB (Oxford Centre for Functional Magnetic Resonance
    Imaging of the Brain), Department of Clinical Neurology, Oxford
    University, Oxford, UK


    LICENCE

    FMRIB Software Library, Release 6.0 (c) 2018, The University of
    Oxford (the "Software")

    The Software remains the property of the Oxford University Innovation
    ("the University").

    The Software is distributed "AS IS" under this Licence solely for
    non-commercial use in the hope that it will be useful, but in order
    that the University as a charitable foundation protects its assets for
    the benefit of its educational and research purposes, the University
    makes clear that no condition is made or to be implied, nor is any
    warranty given or to be implied, as to the accuracy of the Software,
    or that it will be suitable for any particular purpose or for use
    under any specific conditions. Furthermore, the University disclaims
    all responsibility for the use which is made of the Software. It
    further disclaims any liability for the outcomes arising from using
    the Software.

    The Licensee agrees to indemnify the University and hold the
    University harmless from and against any and all claims, damages and
    liabilities asserted by third parties (including claims for
    negligence) which arise directly or indirectly from the use of the
    Software or the sale of any products based on the Software.

    No part of the Software may be reproduced, modified, transmitted or
    transferred in any form or by any means, electronic or mechanical,
    without the express permission of the University. The permission of
    the University is not required if the said reproduction, modification,
    transmission or transference is done without financial return, the
    conditions of this Licence are imposed upon the receiver of the
    product, and all original and amended source code is included in any
    transmitted product. You may be held legally responsible for any
    copyright infringement that is caused or encouraged by your failure to
    abide by these terms and conditions.

    You are not permitted under this Licence to use this Software
    commercially. Use for which any financial return is received shall be
    defined as commercial use, and includes (1) integration of all or part
    of the source code or the Software into a product for sale or license
    by or on behalf of Licensee to third parties or (2) use of the
    Software or any derivative of it for research with the final aim of
    developing software products for sale or license to a third party or
    (3) use of the Software or any derivative of it for research with the
    final aim of developing non-software products for sale or license to a
    third party, or (4) use of the Software to provide any service to an
    external organisation for which payment is received. If you are
    interested in using the Software commercially, please contact Oxford
    University Innovation ("OUI"), the technology transfer company of the
    University, to negotiate a licence. Contact details are:
    fsl@innovation.ox.ac.uk quoting Reference Project 9564, FSL.*/

#include "profilemodel.h"
#include <boost/make_shared.hpp>
#include <boost/log/trivial.hpp>

ProfileModel::ProfileModel(const std::vector<std::string> &modalitynames, int reflength, int datalength)
    : m_modalityNames(modalitynames),
      m_refLength(reflength),
      m_dataLength(datalength)
{
    BOOST_LOG_TRIVIAL(debug) << "Initialising base ProfileModel";

    for (auto mn : modalitynames)
    {
        m_globalMean[mn] = 0.0;
        m_globalStdev[mn] = 1.0;
    }
}

ProfileModel::~ProfileModel()
{
}

int ProfileModel::GetNumberOfSteps() const
{
    return m_refLength - m_dataLength;
}

void ProfileModel::SetNormalisation(const std::string &modality, double mean, double stdev)
{
    if (m_globalMean.find(modality) == m_globalMean.end())
        throw ModelException("Specified modality does not exist in model");

    m_globalMean[modality] = mean;

    // This scaling constant changes the scaling between the mean and covcoefs parts of the optimization parameters
    // as covcoefs is a squared quantity .. it is arbitrary, but helps a lot for numerical stability
    BOOST_LOG_TRIVIAL(debug) << "Note: Scaling intensities by 1000 for optimisation";
    m_globalStdev[modality] = 0.001 * stdev;
}
