
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_xc_pw6b95_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_pw6b95", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-5.279245595407287e-01, -4.102641410721832e-01, -2.571947751373393e-01, -2.318157875484801e-01, -1.214619010253652e+01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_xc_pw6b95_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_pw6b95", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-5.474911863858443e-01, -2.637135282942888e-01, -5.302336047071546e-01, -2.516794582420745e-01, -3.011422610004255e-01, -1.959596064000406e-01, -1.971554137466691e-03, -7.076832852910915e-03, 4.750767786367155e+00, -5.521816590639246e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_pw6b95_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_pw6b95", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.147719098306384e-03, 0.000000000000000e+00, 1.470304759124597e+20, 9.035621880991973e-03, 0.000000000000000e+00, 1.403830521259919e+20, 3.528157083691424e-01, 0.000000000000000e+00, 1.095616217477340e+20, 1.763487874853040e+03, 0.000000000000000e+00, 2.398097476614318e+16, 5.354365166476652e+11, 0.000000000000000e+00, 1.871601652797520e+15]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_pw6b95_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_pw6b95", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-1.575097548581454e-02, 0.000000000000000e+00, -1.906745283611571e-02, 0.000000000000000e+00, -4.501873531899959e-02, 0.000000000000000e+00, -3.326396544035907e-01, 0.000000000000000e+00, -1.772739378606671e+01, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
