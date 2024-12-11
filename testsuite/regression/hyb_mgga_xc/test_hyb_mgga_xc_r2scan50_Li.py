
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_xc_r2scan50_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_r2scan50", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.027664187464626e+00, -7.111903575517674e-01, -1.821304081896372e-01, -9.277703725600933e-02, -4.141097551027171e-02, -2.620751165313680e-03, -5.371020912182091e-07]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_xc_r2scan50_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_r2scan50", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.392202470252032e+00, -1.393322997088805e+00, -9.776475668845757e-01, -9.781889731242812e-01, -2.309123181326433e-01, -2.312185000577526e-01, -1.219522541926516e-01, -2.128626724649335e-03, -5.087448195884754e-02, -5.551370517354495e-02, 9.075780954824922e-02, 1.604708783214137e-01, 5.074097489851123e-05, -1.610477432387998e-07]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_r2scan50_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_r2scan50", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.047098392437207e-04, 1.420514145497902e-04, -1.040326420421731e-04, -3.385720786495147e-04, 1.115362495104119e-03, -3.352897766262414e-04, 1.995171041426807e-01, 4.180228328436082e-01, 1.992400223637515e-01, 1.203693384394904e+00, 7.035114068157315e+00, -4.097406414750609e+03, 1.174424729718431e+02, 2.641353630163130e+02, -3.543765005021111e+06, -4.257954424921900e+01, 8.575971354726938e-01, -3.665958935194394e+03, -2.401652077601736e+00, 4.666265396414997e-04, 2.333132698207498e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_r2scan50_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_r2scan50", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [4.831406615621505e-03, 4.820892587412764e-03, 6.377056619114739e-03, 6.359958047994805e-03, -8.976610258049902e-05, -2.477814696141985e-05, -3.223819977805611e-02, -6.837476715043227e-02, -1.885031156977729e-02, -4.767734640579578e-02, 6.289664780784506e-04, 5.344174127866363e-02, 2.927994619822703e-10, -6.324714924461792e-15]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
