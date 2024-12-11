
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_bcgp_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_bcgp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.801845142971644e+00, -1.293156238866647e+00, -4.232754336767574e-01, -1.604319918000603e-01, -8.198860695343792e-02, -2.054595212136102e-02, -3.838587202672339e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_bcgp_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_bcgp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.234434840868734e+00, -2.236571723870413e+00, -1.510616759762293e+00, -1.511982146604556e+00, -4.126925583306000e-01, -4.128950725330600e-01, -2.048241155899875e-01, -2.612086393062142e-02, -7.743374434527453e-02, -8.296437354427622e-04, -2.746325152792056e-02, -2.726562968526569e-02, -5.541556341433909e-04, -3.939542469250965e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_bcgp_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_bcgp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-2.862026492464950e-04, 0.000000000000000e+00, -2.852224115930352e-04, -1.121729361985634e-03, 0.000000000000000e+00, -1.118156027123166e-03, -7.355964930024342e-02, 0.000000000000000e+00, -7.336064154054485e-02, -4.451093324865353e+00, 0.000000000000000e+00, -2.449030604047108e-01, -6.977052723643381e+01, 0.000000000000000e+00, -1.566097403523873e+00, -2.488820711668978e-01, 0.000000000000000e+00, -2.324083300485849e-01, -1.140060643368612e+00, 0.000000000000000e+00, -1.631879333083259e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
