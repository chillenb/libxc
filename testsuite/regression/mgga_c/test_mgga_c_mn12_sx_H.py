
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_mn12_sx_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_mn12_sx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-4.697412175717099e-03, -2.049870640256986e-02, -3.325276635740315e-02, -2.311487084521242e-01, -3.258401841444964e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_mn12_sx_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_mn12_sx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [1.720003291801302e-02, -5.388092306577726e+02, -6.426688750581033e-02, -2.059476100616399e-01, 6.070266871465532e-02, -1.589949448745367e-01, -1.974151975729120e-01, -1.500178788001988e+00, -4.153512720892640e-02, -1.467647494302934e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_mn12_sx_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_mn12_sx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-4.761449747907557e-01, -9.522899495815114e-01, -4.761449747907557e-01, -4.967998142849421e-02, -9.935996285698842e-02, -4.967998142849421e-02, 5.141811849978926e-01, 1.028362369995785e+00, 5.141811849978926e-01, 4.165725641495049e+02, 8.331451282990098e+02, 4.165725641495049e+02, 1.002100695881174e+06, 2.004201391762349e+06, 1.002100695881174e+06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_mn12_sx_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_mn12_sx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [3.589955636064454e-01, 3.588146565314151e-01, 5.264252963506782e-02, 5.228978033128869e-02, -1.185585595521579e-01, -1.183203110109956e-01, -9.003763268050058e-02, -9.003709335742578e-02, -1.159898547428992e-05, -1.159898554086024e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
