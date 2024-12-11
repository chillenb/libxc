
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_revscan_vv10_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_revscan_vv10", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-7.702172233337023e-16, -1.140993026158950e-02, -2.518301736579695e-02, -2.900405977819791e-02, -3.719930958628729e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_revscan_vv10_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_revscan_vv10", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.829013610409463e-04, -2.294040732595017e-01, 8.220080687792741e-03, -2.346695001191782e-01, -2.840565944235601e-02, -1.951186669235185e-01, -3.022626279347794e-02, -5.218870349835685e-02, -4.742552045685005e-03, -1.589392294537759e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_revscan_vv10_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_revscan_vv10", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [7.481249576060315e-03, 1.496249915212063e-02, 7.481249576060315e-03, 2.455583328732964e-02, 4.911166657465928e-02, 2.455583328732964e-02, 2.008076491196637e-01, 4.016152982393273e-01, 2.008076491196637e-01, 7.390279390283084e+01, 1.478055878056617e+02, 7.390279390283084e+01, 1.222284958553580e+07, 2.444569917107160e+07, 1.222284958553580e+07]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_revscan_vv10_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_revscan_vv10", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-1.784266505930125e-02, -1.763348632679498e-02, -2.690912684087913e-02, -2.647047775884577e-02, -3.774396171027058e-04, -3.760682128707441e-04, -4.860512479216227e-03, -4.860386565833174e-03, -6.899944337691533e-07, -6.899944391487235e-07]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
