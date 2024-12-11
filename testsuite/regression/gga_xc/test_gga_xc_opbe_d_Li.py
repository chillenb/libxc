
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_xc_opbe_d_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_opbe_d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.860125286158627e+00, -1.334062816034628e+00, -4.461662578478289e-01, -1.759359964387761e-01, -8.469129102364464e-02, -2.505217342201708e-02, -4.683329389424264e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_xc_opbe_d_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_opbe_d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.359153927959459e+00, -2.361158831813047e+00, -1.619268152213839e+00, -1.620541812773554e+00, -3.850994124240580e-01, -3.852801344436959e-01, -2.288079054493962e-01, -1.377910991984524e-01, -8.018601348634430e-02, 4.636626615963091e-01, -3.344444387351375e-02, -3.320674258401173e-02, -6.761056351412147e-04, -4.806499136599841e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_xc_opbe_d_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_opbe_d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-2.134202666173689e-04, 8.082590462822176e-05, -2.125327328138019e-04, -8.963087691228003e-04, 2.753884844702505e-04, -8.929257648376244e-04, -1.056931415311627e-01, 9.683278318315822e-03, -1.055121314881578e-01, -1.376064435441974e+00, 5.012185980240757e+00, 1.865244615945281e+00, -7.036837098833432e+01, 3.169441818943081e+01, 1.174237772289372e+01, -6.507896061095647e-01, 6.949247971220851e-04, -6.077435638860271e-01, -2.988168969322332e+00, 6.656123155997806e-06, -4.277260103884513e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
