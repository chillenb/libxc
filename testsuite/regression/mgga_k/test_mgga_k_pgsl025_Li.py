
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_k_pgsl025_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_pgsl025", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [1.674466329113066e+01, 8.848660260075597e+00, 3.216440028534375e+00, 1.340409731925616e-01, "nan", inf, inf]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_k_pgsl025_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_pgsl025", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [2.452947926179061e+01, 2.457941459354898e+01, 9.838012270656755e+00, 9.860297692641236e+00, -3.198076218303156e+00, -3.207946612931091e+00, 2.097132137530909e-01, -3.057065625943076e+00, "nan", "nan", "nan", "nan", "nan", "nan"]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_k_pgsl025_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_pgsl025", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [5.749160568948524e-03, 0.000000000000000e+00, 5.727837265760318e-03, 2.580945326420985e-02, 0.000000000000000e+00, 2.572594671925097e-02, 4.147700292392334e+00, 0.000000000000000e+00, 4.153136729091924e+00, 5.871854119231491e+00, 0.000000000000000e+00, 7.829811571772351e+04, 4.018360953043848e+02, 0.000000000000000e+00, 2.454376727176486e+09, 6.733474755479385e+04, 0.000000000000000e+00, 6.882557364072798e+04, 8.236143114997760e+09, 0.000000000000000e+00, 2.292358375079883e+10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_k_pgsl025_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_pgsl025", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [5.470907259762659e-05, 1.887559832251140e-314, 1.175276021155877e-313, 1.170287128969648e-313, 5.434722104253712e-323, 3.606679214641100e-321, 0.000000000000000e+00, 1.961401495975017e-303, 7.664054992731501e-318, "nan", 1.115379825624925e-315, 5.919462327249798e+167, 1.539541104002989e-137, 4.014634969612438e+263]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_k_pgsl025_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_pgsl025", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
