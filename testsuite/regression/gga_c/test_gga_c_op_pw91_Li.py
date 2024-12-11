
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_op_pw91_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_op_pw91", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-5.381448231968380e-02, -4.627128190979510e-02, -1.311714598421946e-02, -1.699107586311927e+00, -8.877352693212557e-10, -2.985758446342468e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_op_pw91_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_op_pw91", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-7.014943603354540e-02, -6.997192518657899e-02, -6.887216538193133e-02, -6.869923593018019e-02, -2.896548484656191e-02, -2.899553147523454e-02, -1.698059782698428e+00, 2.640335383879315e+04, -1.001076571046833e-09, -2.653748015581691e-03, 1.182646894250651e+01, 1.315982629393839e+01, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_op_pw91_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_op_pw91", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([2.322980268714172e-05, 0.000000000000000e+00, 2.305195989808155e-05, 1.024742907317653e-04, 0.000000000000000e+00, 1.017347260887566e-04, 6.741619421129274e-03, 0.000000000000000e+00, 6.734439492971488e-03, 1.594818655805294e-01, 0.000000000000000e+00, -2.699848597738179e+08, 1.740694490489732e-06, 0.000000000000000e+00, 0.000000000000000e+00, -1.317481127847526e+05, 0.000000000000000e+00, -1.408609884277317e+05, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
