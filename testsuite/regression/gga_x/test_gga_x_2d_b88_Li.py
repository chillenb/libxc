
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_2d_b88_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_2d_b88", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-3.911513566737549e+00, -2.325519582615999e+00, -3.963996001024804e-01, -1.132423307659920e-01, -4.540466555149302e-02, -7.176648990972942e-02, -2.845462609622332e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_2d_b88_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_2d_b88", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-5.662119587688783e+00, -5.669927293987840e+00, -3.237260223828666e+00, -3.241465734491750e+00, -4.083529900082496e-01, -4.081790173105227e-01, -1.511870044430038e-01, -1.432309436697474e-02, -4.254690361408922e-02, -3.134122274337332e-03, -1.469951082384231e-02, -1.479468880848459e-02, -3.070304227635527e-03, -2.646697302620789e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_2d_b88_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_2d_b88", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-3.066500647135411e-04, 0.000000000000000e+00, -3.055931382838224e-04, -1.165664600697818e-03, 0.000000000000000e+00, -1.162051836685058e-03, -8.029968956793672e-02, 0.000000000000000e+00, -8.024519465340554e-02, -8.087061353493215e+00, 0.000000000000000e+00, -7.845942871141833e+02, -4.971684237940597e+01, 0.000000000000000e+00, -2.672526713881112e+07, -6.828051008362560e+02, 0.000000000000000e+00, -6.841629359679107e+02, -7.918602442536075e+07, 0.000000000000000e+00, -2.346894226328627e+08]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
