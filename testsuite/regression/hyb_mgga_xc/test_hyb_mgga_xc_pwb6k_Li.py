
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_xc_pwb6k_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_pwb6k", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.026601586089037e+00, -7.445139857111680e-01, -2.420469869241837e-01, -8.728193690715962e-02, -4.478449529108966e-02, -1.414418202053475e-04, -9.600636696070276e-07])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_xc_pwb6k_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_pwb6k", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.267795494627979e+00, -1.268733502932569e+00, -8.800957544385721e-01, -8.806312282608142e-01, -2.436419880185016e-01, -2.436934816466243e-01, -1.121760110965552e-01, -1.182189150891131e-03, -4.280673523579744e-02, -1.079246196008839e-06, -6.581057514466445e-04, -5.423260081243179e-04, -6.500451792818399e-06, -3.055161806128015e-09])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_pwb6k_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_pwb6k", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.363339012072015e-04, 0.000000000000000e+00, -1.359522907179234e-04, -4.868240707370991e-04, 0.000000000000000e+00, -4.856716807691699e-04, -3.650639601805085e-02, 0.000000000000000e+00, -3.646287325819621e-02, -4.876020110929686e-01, 0.000000000000000e+00, 8.087115996685830e+00, -3.343350483424010e+01, 0.000000000000000e+00, 5.962477020028172e+02, 4.027913559198457e+00, 0.000000000000000e+00, 3.052172634753293e+00, 1.514295820929593e+04, 0.000000000000000e+00, 1.504574279092344e+01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_pwb6k_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_pwb6k", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-1.468068417266649e-03, -1.467344030576164e-03, -1.635145409581866e-03, -1.634518918973160e-03, -3.808753784958722e-04, -3.784621145227853e-04, -9.162600290322200e-02, -1.000804921357904e-07, -1.362420881216888e-02, -3.279466530104145e-11, -1.168130236790750e-07, -1.069038987461692e-07, -7.274709012627254e-12, -3.801003672837534e-12])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
