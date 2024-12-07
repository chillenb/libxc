
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_mggac_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mggac", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.038312134688814e+00, -1.415145164630983e+00, -3.373070552922985e-01, -1.841893665828348e-01, -7.304752499715957e-02, -1.200772215948481e-02, -1.999502583511418e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_mggac_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mggac", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.710322637373699e+00, -2.712820461167592e+00, -1.877110637584933e+00, -1.887669587692605e+00, -1.661602156059725e-01, -4.529004471021008e-01, -2.452379677799075e-01, -1.702440202814497e-02, -9.447781100395083e-02, 3.318515358245124e+01, -1.428664383489929e-02, -1.777211126097943e-02, -2.878074302689367e-04, 2.502692401136107e+01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mggac_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mggac", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.835222673925526e-05, 0.000000000000000e+00, -2.804291936769523e-05, -1.258480577941772e-04, 0.000000000000000e+00, 0.000000000000000e+00, -3.808786493859629e-01, 0.000000000000000e+00, 0.000000000000000e+00, -6.005608409561236e-01, 0.000000000000000e+00, 0.000000000000000e+00, -1.703265836133783e+01, 0.000000000000000e+00, -6.730156283022871e+10, -4.068678517035588e-12, 0.000000000000000e+00, 0.000000000000000e+00, -2.776248632682398e-24, 0.000000000000000e+00, -4.839054093056178e+11]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mggac_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mggac", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mggac_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mggac", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [1.471166793223804e-03, 1.459054008205392e-03, 2.185491278919759e-03, 2.101601670246545e-13, 9.170270904690263e-02, 8.759385714450331e-13, 2.305387138628149e-02, 2.330323742747301e-11, 4.073319581179562e-02, 2.742104014866497e+01, 6.042464943069069e-17, 2.232278728590314e-11, 3.368752375220091e-34, 2.110950079037119e+01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
