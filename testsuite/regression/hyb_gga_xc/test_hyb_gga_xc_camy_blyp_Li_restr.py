
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_camy_blyp_Li_restr_1_zk():
    # Prepare the input
    inp = test_data["Li_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_camy_blyp", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.332986399468061e+00, -9.173657425909169e-01, -1.494913314657606e-01, -4.468947765958166e-02, -9.788146166366992e-03, -2.126057103621954e-03, -3.861808238418421e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_camy_blyp_Li_restr_1_vrho():
    # Prepare the input
    inp = test_data["Li_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_camy_blyp", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.709871318288987e+00, -1.148640644899350e+00, -2.980805318217697e-01, -6.878851322407543e-02, -1.692059960576814e-02, -2.827930883169290e-03, -5.148084704558810e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_camy_blyp_Li_restr_1_vsigma():
    # Prepare the input
    inp = test_data["Li_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_camy_blyp", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-8.674470304568186e-05, -2.881824688462315e-04, 1.466654814355409e-02, 3.382641428442269e-01, 8.174293496635139e+00, 4.007944914529413e-02, -9.126568925047720e-09]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
