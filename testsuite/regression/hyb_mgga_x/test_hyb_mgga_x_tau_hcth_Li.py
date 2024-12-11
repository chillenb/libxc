
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_x_tau_hcth_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_tau_hcth", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.504835656782309e+00, -1.044779895773110e+00, -2.510846032747648e-01, -1.360340934690236e-01, -5.400322647966888e-02, -2.898880583316085e-02, -4.691314931259089e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_x_tau_hcth_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_tau_hcth", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.006501961566627e+00, -2.008337773407269e+00, -1.391403902091293e+00, -1.392594205829885e+00, -3.350330167764849e-01, -3.348972431523953e-01, -1.815379174548105e-01, -3.672928602166876e-02, -7.203563120574237e-02, -1.171032597310078e-03, -3.882306259998533e-02, -3.832997588001615e-02, -7.821930148474499e-04, -1.894104816437915e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_tau_hcth_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_tau_hcth", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-7.023010436823965e-05, 0.000000000000000e+00, -6.992840900037586e-05, -3.043937357545704e-04, 0.000000000000000e+00, -3.031485290849473e-04, -1.345597498136940e-01, 0.000000000000000e+00, -1.352514232394815e-01, -1.164767842073477e+00, 0.000000000000000e+00, -1.244966432321287e+00, -7.006917142501514e+01, 0.000000000000000e+00, -7.977681076742294e+00, -5.313822926352427e-04, 0.000000000000000e+00, -1.181299432509990e+00, -3.641287287666552e-10, 0.000000000000000e+00, -1.166530809982177e+12]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_tau_hcth_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_tau_hcth", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [1.426771708980569e-04, 1.424714475178577e-04, -1.525732882708735e-04, -1.516396420757867e-04, 3.346256905451419e-05, 3.487754658197196e-05, 3.159118677151351e-03, 2.710184102798595e-08, 2.622176468528291e-04, 1.412260888820887e-14, 3.075400113432719e-13, 3.105391039183646e-08, -9.529891000228186e-24, 1.440590437500646e-19]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
