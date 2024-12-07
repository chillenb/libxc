
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_x_revm06_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_revm06", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-9.436448351346061e-01, -7.386885532865695e-01, -2.433250593732508e-01, -8.179852687541410e-02, -5.032434812009874e-02, -3.714146509795919e-02, -6.957838331239461e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_x_revm06_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_revm06", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-9.383612797329529e-01, -9.395734416541163e-01, -7.720508225287733e-01, -7.724254071200550e-01, -2.528291162895464e-01, -2.577965777855214e-01, -1.142483519993746e-01, -4.681237847836917e-02, -4.398925611409275e-02, -1.503773137099468e-03, -4.974028726658768e-02, -4.882971907006512e-02, -1.004465130958499e-03, -7.140787381110401e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_revm06_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_revm06", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.422862630704853e-04, 0.000000000000000e+00, -1.417378775436825e-04, -6.323514723813977e-04, 0.000000000000000e+00, -6.301186709715442e-04, -7.386625501749430e-02, 0.000000000000000e+00, -7.338711791134889e-02, -2.115154822250216e+00, 0.000000000000000e+00, -9.594334850522954e-01, -5.741037725095954e+01, 0.000000000000000e+00, -6.164879210406838e+00, -5.298872503081524e-01, 0.000000000000000e+00, -9.102141267466308e-01, -2.344046520634770e+00, 0.000000000000000e+00, -6.423887884637646e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_revm06_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_revm06", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_revm06_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_revm06", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-3.362765356116619e-02, -3.357472255338410e-02, -1.352948275440367e-02, -1.357865962240999e-02, 1.481675479805952e-02, 1.565527505423697e-02, 2.705545591048785e-01, -4.940325382679366e-05, 2.617447696588302e-02, -1.012454770879636e-08, -5.741078633118399e-08, -5.332018560703878e-05, -3.884298007947035e-19, -1.129554862957343e-09]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
