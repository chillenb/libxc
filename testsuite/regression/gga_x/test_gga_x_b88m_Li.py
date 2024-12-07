
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_b88m_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_b88m", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.806548847707896e+00, -1.295672318220003e+00, -4.362276423346982e-01, -1.608261739518918e-01, -8.195840920201838e-02, -1.333226804661893e-01, -5.361494935528503e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_b88m_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_b88m", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.239080027379213e+00, -2.241199494007153e+00, -1.522142949061213e+00, -1.523500958966748e+00, -3.426130946717470e-01, -3.424959985217574e-01, -2.048451036587874e-01, -3.600704817022713e-02, -7.359324009240371e-02, -7.727024854237556e-03, -3.686432373956571e-02, -3.703763262131878e-02, -7.462904379212655e-03, -6.454223130829584e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_b88m_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_b88m", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.889861169776130e-04, 0.000000000000000e+00, -2.880353003775789e-04, -1.078659546335521e-03, 0.000000000000000e+00, -1.075297601733299e-03, -1.158906391247879e-01, 0.000000000000000e+00, -1.158519110797816e-01, -4.672532468226923e+00, 0.000000000000000e+00, -1.339213351894191e+03, -7.808631205290501e+01, 0.000000000000000e+00, -4.850267518446801e+07, -1.164518094107250e+03, 0.000000000000000e+00, -1.166386238603503e+03, -1.439996305197792e+08, 0.000000000000000e+00, -4.289598160291446e+08]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
