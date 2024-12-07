
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_2d_b86_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_2d_b86", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-3.873778571716186e+00, -2.284436895526075e+00, -6.632071269140596e-01, -1.110925380160329e-01, -1.066754800724844e-01, -3.604399085869291e-02, -9.270118292318955e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_2d_b86_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_2d_b86", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-5.688213966463441e+00, -5.696064676593443e+00, -3.225785896868142e+00, -3.230104907794371e+00, 9.985224452197565e-02, 1.012581318609556e-01, -1.460582903122409e-01, -5.036111574040546e-02, 3.701271674024977e-02, -2.848882474203327e-04, -5.429191794070529e-02, -5.370603246157944e-02, -1.555189244958608e-04, -9.321889216757116e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_2d_b86_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_2d_b86", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.806585040688562e-04, 0.000000000000000e+00, -1.799284192632086e-04, -9.308811806882894e-04, 0.000000000000000e+00, -9.273308304172702e-04, -4.716175379303471e-01, 0.000000000000000e+00, -4.721495900194395e-01, -8.918152091194418e+00, 0.000000000000000e+00, -4.452384233285978e-01, -3.832343834751207e+02, 0.000000000000000e+00, -1.606384598074579e-02, -4.879498920025590e-01, 0.000000000000000e+00, -4.506692453059054e-01, -6.383603642916682e-03, 0.000000000000000e+00, -5.477045313670408e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
