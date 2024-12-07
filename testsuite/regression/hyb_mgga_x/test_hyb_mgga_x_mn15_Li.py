
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_x_mn15_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_mn15", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [9.071253858470841e-03, -5.544711986122137e-01, -4.686695950164392e-01, 1.117063790917585e-01, -7.188290250219971e-02, 1.431154910788963e-02, 3.384121619431041e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_x_mn15_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_mn15", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [2.601285435371800e+00, 2.609432248110288e+00, 3.330820435260930e-01, 3.374150195072313e-01, -1.843117226707288e-01, -1.828145620012169e-01, 3.234046779891407e-01, 1.710243469682537e-02, -2.906699232999999e-02, 7.284341260648457e-04, 1.792447345172004e-02, 1.757992350004363e-02, 4.878897199940569e-04, 3.473898760176992e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_mn15_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_mn15", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.191334909413866e-03, 0.000000000000000e+00, -2.188351558064914e-03, -3.548120275411985e-03, 0.000000000000000e+00, -3.552032290709731e-03, -2.324867220805285e-01, 0.000000000000000e+00, -2.295370891127956e-01, -5.422037085514933e+00, 0.000000000000000e+00, 9.346469625220219e-03, -2.421957986475181e+02, 0.000000000000000e+00, 1.463761551925373e+00, -5.796754775078263e-04, 0.000000000000000e+00, -2.812562698136375e-04, 1.077035715968060e+00, 0.000000000000000e+00, 1.551205072680463e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_mn15_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_mn15", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_mn15_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_mn15", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-1.550013120549297e-01, -1.553326590174859e-01, -5.178933343646332e-02, -5.191676078408072e-02, -7.714931935425724e-04, -1.805863627291106e-03, -6.163706201848801e+00, 2.632897231497640e-05, 3.659233641414082e-01, 5.712495257468468e-09, 1.299186997155816e-08, 2.834408246321334e-05, 7.774190989830327e-20, 6.379796061438055e-10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
