
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_mpw3lyp_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_mpw3lyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.468326401303264e+00, -1.060372534487472e+00, -3.203077953082270e-01, -1.298148787276074e-01, -6.513437264546035e-02, -5.661224018989558e-03, -1.149825824212573e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_mpw3lyp_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_mpw3lyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.830343862671427e+00, -1.831825764887131e+00, -1.269715959342838e+00, -1.270610182925394e+00, -3.834313443413655e-01, -3.837842923241794e-01, -1.656758815793477e-01, -9.281414304899162e-02, -6.254562095738621e-02, -3.650677275321516e-02, -1.021853423651188e-02, -1.007575649746113e-02, -1.397880926587230e-04, -1.830498219856196e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_mpw3lyp_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_mpw3lyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.899791995698407e-04, 4.549072232432841e-06, -1.894417178221994e-04, -6.973399297093677e-04, 3.176486298435520e-05, -6.956235961019958e-04, -4.213320215297581e-02, 4.157947454183569e-02, -4.193268454728159e-02, -3.159187061662803e+00, 4.003233384193598e+00, 2.725928895479368e+01, -5.184485749527933e+01, 2.052894508601134e+01, 3.422389238078151e+02, 2.435559428908481e+01, 6.912340767268339e-02, 2.288483936216668e+01, 2.714462259467661e+02, 0.000000000000000e+00, 4.163649100840559e+02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
