
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_zpbesol_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_zpbesol", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-6.703038854967247e-02, -5.108206521841256e-02, -5.363806991765085e-03, -1.607161149585005e-02, -7.369504650861261e-03, -4.290713484707909e-04, -1.681738173367795e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_zpbesol_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_zpbesol", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.172911624863344e-01, -1.172219625438186e-01, -1.053537628477081e-01, -1.053762627817033e-01, -2.573715285085386e-02, -2.523427852643498e-02, -2.314743833632123e-02, -1.021828539409714e-01, -5.062045356749381e-04, 2.961198693911313e+00, -7.410051704422098e-02, 7.758262288615372e-02, -1.978391892536515e-04, -2.903012928114558e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_zpbesol_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_zpbesol", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([4.028535930565969e-05, 8.057071861131940e-05, 4.028535930565969e-05, 1.373916747477152e-04, 2.747833494954304e-04, 1.373916747477152e-04, 4.882058283754389e-03, 9.764116567508783e-03, 4.882058283754389e-03, 2.285184831532090e+00, 4.570369663064181e+00, 2.285184831532090e+00, -2.142307653071803e+01, -4.284615306143604e+01, -2.142307653071803e+01, -6.872222946145794e+00, -1.374444589229157e+01, -6.872222946145794e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
