
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_lc_bop_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_lc_bop", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.591590813021817e+00, -1.067274337027997e+00, -1.399694593430772e-01, -2.647729353660904e-02, -2.053077997311435e-03, -2.566514009077260e-05, -5.172070017611084e-10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_lc_bop_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_lc_bop", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.077365375045992e+00, -2.079269961676821e+00, -1.375837435778218e+00, -1.376987129662425e+00, -2.245649495768429e-01, -2.245285417908847e-01, -4.864089285069102e-02, -1.046406818475598e-01, -4.029071711256568e-03, -4.228443824848908e-02, -7.186555725155630e-05, -7.210635671912551e-05, -1.297607887759795e-09, -3.265179364717432e-09]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_lc_bop_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_lc_bop", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.912889157775706e-04, 0.000000000000000e+00, -1.908089275585867e-04, -6.126215638018391e-04, 0.000000000000000e+00, -6.113727392895725e-04, -5.317044913970103e-03, 0.000000000000000e+00, -5.284877526707610e-03, -1.630594947284433e-01, 0.000000000000000e+00, 6.990026009826852e+02, -6.620288321695750e-02, 0.000000000000000e+00, 2.343449931948338e+07, 2.497469957014167e-01, 0.000000000000000e+00, 2.495762430321545e-01, 1.510099786433184e+00, 0.000000000000000e+00, 1.575121740585954e+01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
