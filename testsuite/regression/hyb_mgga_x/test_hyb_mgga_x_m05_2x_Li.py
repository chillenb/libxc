
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_x_m05_2x_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m05_2x", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-6.167280152242104e-01, -4.826770995691485e-01, -2.618005426467487e-01, -6.001875030938295e-02, -4.288395607271413e-02, -3.397155584679086e-02, -6.357008037670880e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_x_m05_2x_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m05_2x", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-6.671774804776899e-01, -6.696682227511900e-01, -5.027326864561532e-01, -5.037390099582917e-01, -1.664966788565310e-01, -1.693866298391625e-01, -1.198894730339805e-01, -4.297287857980580e-02, -1.445834442749887e-02, -1.373933976683732e-03, -4.546475539729315e-02, -4.483786864369479e-02, -9.177263311706507e-04, -6.524175414265879e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_m05_2x_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m05_2x", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-8.770509015039491e-05, 0.000000000000000e+00, -8.738266882252321e-05, -3.800135456872690e-04, 0.000000000000000e+00, -3.787141327973936e-04, -4.709749220404015e-02, 0.000000000000000e+00, -4.676086668865988e-02, -1.481333297492442e+00, 0.000000000000000e+00, -4.586118502742421e-01, -3.606191352859302e+01, 0.000000000000000e+00, -2.941927861630426e+00, -4.673472958659297e-01, 0.000000000000000e+00, -4.351295303387527e-01, -2.141630537713454e+00, 0.000000000000000e+00, -3.065518662983593e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_m05_2x_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m05_2x", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-1.474467967368138e-02, -1.447689928825859e-02, -1.001704166862535e-02, -9.935390234032147e-03, -1.567363150128290e-02, -1.533496776613043e-02, 1.220382671079884e+00, -3.018231609631271e-05, -2.197096149579273e-01, -6.168896743427215e-09, -1.496852473650731e-08, -3.258072019469985e-05, -8.399670313978482e-20, -6.882352179445759e-10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
