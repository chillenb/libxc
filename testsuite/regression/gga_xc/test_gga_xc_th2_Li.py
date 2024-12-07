
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_xc_th2_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_th2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.933357282723467e+00, -1.335843031250463e+00, -4.779388274108832e-01, -1.516460072609786e-01, -6.751554778006399e-02, -1.877624121741872e-01, -9.909575133058497e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_xc_th2_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_th2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.661579319572791e+00, -2.664774250011748e+00, -1.717622326925892e+00, -1.719543345572024e+00, -2.765687934930911e-01, -2.764089199905101e-01, -1.902254451398532e-01, -8.219811921281450e-02, -7.220425802012335e-02, 7.918585611113951e-02, 1.797603963850839e-01, 1.850928522492953e-01, 8.029147449265579e-02, 1.362860688681249e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_xc_th2_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_th2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [8.878172726642096e-04, -1.548424656946587e-03, 8.866757229140788e-04, 6.851598649434995e-04, -1.999962057807564e-03, 6.825226497840437e-04, -1.281585071681732e-01, -9.960159765289403e-02, -1.281528916789845e-01, -6.140678393976221e+00, -3.742339171458382e+00, -1.140668403402440e+03, -5.776941011800422e+01, -1.030148482166434e+02, -7.194260130422613e+07, -1.578766324634411e+02, -8.101334520887890e+03, -1.622520209160015e+02, 2.528238911427956e+08, -2.056335139663752e+09, 8.533892880376904e+07]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
