
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_xc_b98_H_restr_1_zk():
    # Prepare the input
    inp = test_data["H_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_b98", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.911461163649500e-01, -3.436033418983573e-01, -2.518245981152757e-01, -1.073712649401645e-01, -5.152374649782685e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_xc_b98_H_restr_1_vrho():
    # Prepare the input
    inp = test_data["H_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_b98", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-4.583372911537693e-01, -5.882526832294603e-01, -3.138560225361988e-01, -1.322298948204771e-01, -7.134567559051113e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_b98_H_restr_1_vsigma():
    # Prepare the input
    inp = test_data["H_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_b98", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [2.583392255524145e+00, 6.407483719201824e-02, -9.316361313239756e-02, -5.601649214109192e-01, 5.576820634589121e+02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_b98_H_restr_1_vlapl():
    # Prepare the input
    inp = test_data["H_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_b98", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [-2.054212429620973e-04, -3.624487054031979e-03, -9.217712217645181e-03, -2.073933858831241e-03, -1.063009166570416e-10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_b98_H_restr_1_vtau():
    # Prepare the input
    inp = test_data["H_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_b98", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-6.161350746628632e+00, -1.101631830668243e-01, 3.205023958618487e-02, 3.670410940187083e-03, -3.820665493825896e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
