
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_xc_hcth_p14_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_hcth_p14", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.907938735518451e+00, -1.355353879253772e+00, -4.408819108177318e-01, -2.083369465247233e-01, -1.000527670216479e-01, -3.203387251791938e-02, -7.098102101635001e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_xc_hcth_p14_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_hcth_p14", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.515656805288099e+00, -2.517957405429928e+00, -1.739341565130236e+00, -1.740874107318660e+00, -4.491917479622106e-01, -4.493957410050013e-01, -2.760195868116036e-01, -1.868940387920578e-01, -1.036468633976701e-01, -9.410139961505619e-02, -4.165444743511260e-02, -4.156854838331434e-02, -9.630226046507496e-04, -8.818789146101976e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_xc_hcth_p14_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_hcth_p14", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([6.607976182611763e-06, 0.000000000000000e+00, 6.872372507461149e-06, -2.059391909897320e-04, 0.000000000000000e+00, -2.043664061302935e-04, -6.054579377286223e-02, 0.000000000000000e+00, -6.036056768461302e-02, 3.042979256251635e+00, 0.000000000000000e+00, -2.897653096052207e+01, -5.658263917045163e+01, 0.000000000000000e+00, -3.364850535197560e+03, -9.128667660982194e-01, 0.000000000000000e+00, -8.963199015210486e-01, -3.197853696063963e+00, 0.000000000000000e+00, -1.264173624684065e+01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
