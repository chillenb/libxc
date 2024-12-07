
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_b97_1p_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b97_1p", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.610705498083760e+00, -1.145836269695564e+00, -3.900764637052360e-01, -1.478009035214955e-01, -7.156183235136297e-02, -1.570829011194159e-02, -3.076762994326867e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_b97_1p_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b97_1p", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.084378800413948e+00, -2.086114187903199e+00, -1.432285378242461e+00, -1.433409235559452e+00, -2.940395485304945e-01, -2.946354003397292e-01, -1.959279754785925e-01, 4.786810001723216e-01, -5.049967762912201e-02, 3.118296186971826e-01, -2.203538012209053e-02, -2.124379351090179e-02, -6.303614984704396e-04, 2.001180750524188e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_b97_1p_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b97_1p", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-7.853550059518788e-05, 0.000000000000000e+00, -7.816567013461639e-05, -4.278702928367401e-04, 0.000000000000000e+00, -4.260650223439832e-04, -1.107538032871895e-01, 0.000000000000000e+00, -1.104688381540281e-01, 1.035220478962333e-01, 0.000000000000000e+00, 7.234703779783352e+01, -9.781388517038008e+01, 0.000000000000000e+00, 8.666776716736502e+03, -3.829744978028536e-01, 0.000000000000000e+00, -2.447633451645141e-01, -4.353571776732271e+00, 0.000000000000000e+00, 1.458131105159143e+01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
