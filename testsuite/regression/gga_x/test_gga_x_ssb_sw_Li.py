
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_ssb_sw_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_ssb_sw", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.831141672859305e+00, -1.286643479472680e+00, -4.161010870154249e-01, -1.651668092476625e-01, -8.052780026924379e-02, -2.054423373337063e-02, -3.838584531770702e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_ssb_sw_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_ssb_sw", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.408289888432536e+00, -2.410578515818145e+00, -1.599749441633265e+00, -1.601314443152526e+00, -4.023693377127568e-01, -4.025467222747406e-01, -2.195151133561448e-01, -2.611487632768952e-02, -7.600944835651292e-02, -8.296427312186548e-04, -2.745623742426715e-02, -2.725899648919734e-02, -5.541551633342622e-04, -3.939539467631104e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_ssb_sw_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_ssb_sw", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-5.508552475647792e-05, 0.000000000000000e+00, -5.473473879492311e-05, -6.062693183864422e-04, 0.000000000000000e+00, -6.034961097131636e-04, -7.392570476070881e-02, 0.000000000000000e+00, -7.374092676278814e-02, -3.755779585554151e-01, 0.000000000000000e+00, -2.831022534678534e-01, -6.862510423038100e+01, 0.000000000000000e+00, -1.811159279861648e+00, -2.876875177657423e-01, 0.000000000000000e+00, -2.686514254917876e-01, -1.318457477021429e+00, 0.000000000000000e+00, -1.887236334936897e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
