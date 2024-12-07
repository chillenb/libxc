
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_xc_tpsslyp1w_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_tpsslyp1w", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.034340730986371e+00, -1.425697692064457e+00, -4.024408943337484e-01, -1.815505725590388e-01, -7.950298992028136e-02, -2.384904318627060e-02, -4.484473413197910e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_xc_tpsslyp1w_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_tpsslyp1w", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.481193976975682e+00, -2.483199996484361e+00, -1.778402341874489e+00, -1.779988371350260e+00, -4.428945708348844e-01, -4.439086755737135e-01, -2.192567702233132e-01, -1.204903826988055e-01, -7.794925082678848e-02, -4.259116523058908e-02, -3.166787726749169e-02, -3.156255780717258e-02, -6.193147838418892e-04, -5.368726579089803e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_xc_tpsslyp1w_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_tpsslyp1w", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-9.853400432551758e-04, 3.864883412170266e-06, -9.845744852398247e-04, -1.552073376279352e-03, 2.698736924043955e-05, -1.549781724533601e-03, -6.423065864684124e-02, 3.532584519053778e-02, -6.345994630413704e-02, -2.805602309204169e+01, 3.401139729395250e+00, 2.273874498182499e+00, -6.386060199994042e+01, 1.744135403403949e+01, 1.130451870105053e+01, -2.523478326694332e-01, 5.872712018115467e-02, -2.334805152723543e-01, -1.293222808933014e+00, 0.000000000000000e+00, -1.851114589210472e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_xc_tpsslyp1w_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_tpsslyp1w", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_xc_tpsslyp1w_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_tpsslyp1w", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [5.133256656990907e-02, 5.146081972621280e-02, 2.699536652654771e-02, 2.707549740077117e-02, 5.249170907102530e-04, 4.408895734294224e-04, 1.012299727590791e+00, 6.714990221518002e-11, 1.276521894098830e-02, 3.628503406176095e-17, 4.835635833592273e-17, 7.697046751037394e-11, 7.924103401730547e-38, 6.041099655890685e-19]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
