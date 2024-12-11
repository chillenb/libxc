
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_x_m08_so_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m08_so", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-2.026453230455548e-01, -3.341139106961014e-01, -1.124452344096898e-01, -4.095234191331540e-02, -2.496596412808236e-02, -1.353528495204439e-02, -2.491874646754534e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_x_m08_so_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m08_so", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-6.887266415333886e-01, -6.983581530843903e-01, 2.994304188259450e-01, 3.031777979673316e-01, -1.771625296969274e-01, -1.781124566320952e-01, -1.282309787637873e-01, -1.733783463405890e-02, -3.262821567311915e-02, -5.453705645879304e-04, -1.808561499993878e-02, -1.810829025648993e-02, -3.642683916915740e-04, -2.431325100467056e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_m08_so_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m08_so", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.211312253306245e-03, 0.000000000000000e+00, -1.208767856727671e-03, -1.429578385688385e-03, 0.000000000000000e+00, -1.438635104493983e-03, -6.136579309757578e-01, 0.000000000000000e+00, -6.151555033933264e-01, -1.805597444355714e+01, 0.000000000000000e+00, 5.870644409002859e-01, -3.106592161197468e+02, 0.000000000000000e+00, 3.821387703712591e+00, 2.544567672841814e-04, 0.000000000000000e+00, 5.564979442535318e-01, 1.744299399878458e-10, 0.000000000000000e+00, 1.103590045713305e+12])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_m08_so_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m08_so", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([5.969307044623650e-02, 6.087710547615484e-02, -1.089813749757453e-01, -1.094534926364348e-01, 4.991049015078898e-03, 5.206619267555631e-03, 2.094316054847225e+00, 5.460765513200752e-06, -5.523999848140422e-03, 1.147231646023665e-09, 2.783845304411809e-09, 5.884569869615716e-06, 1.552088026431582e-20, -4.086487291142884e-10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
