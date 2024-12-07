
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_op_pbe_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_op_pbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-5.395249831655779e-02, -4.650414130126819e-02, -1.323742593227553e-02, -3.875616355655072e-04, -1.065778569411376e-05, -6.655517262649556e-04, -1.075244526060374e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_op_pbe_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_op_pbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-6.951903506811942e-02, -6.934333457192858e-02, -6.916088708104869e-02, -6.898553555330358e-02, -2.769616135133733e-02, -2.772049788778291e-02, -3.628567086622905e-04, -4.497673940119812e-01, -1.066612974419225e-05, -2.111742998744255e+01, -8.717125053408856e-04, -8.875857995958684e-04, -1.056548167354771e-05, -2.482268968993074e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_op_pbe_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_op_pbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [2.189734290119850e-05, 0.000000000000000e+00, 2.172633859706130e-05, 1.026608977819262e-04, 0.000000000000000e+00, 1.019105067312109e-04, 6.056715038800739e-03, 0.000000000000000e+00, 6.048197310127344e-03, 2.168426242047068e-03, 0.000000000000000e+00, 3.053588120887338e+01, 1.376657576048101e-04, 0.000000000000000e+00, 3.549267517993954e+05, 1.836266555886252e-02, 0.000000000000000e+00, 1.778386049932066e-02, 3.780651655501445e-02, 0.000000000000000e+00, 2.980284105431988e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
