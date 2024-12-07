
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_pbe3_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_pbe3", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [2.012184288151019e+00, 1.851442259217876e+00, 9.907733253861030e-01, 1.150692914150202e-01, 2.686153744295622e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_pbe3_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_pbe3", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [3.408712751668945e+00, 3.721490590979981e-16, 1.453777276505223e+00, 1.281124384012476e-16, 6.461066562867810e-01, -8.314504823121616e-18, 1.766307829497432e-01, 7.032321333397596e-17, 4.476311337559178e-04, -3.791028908098392e-20]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_pbe3_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_pbe3", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-8.447368956458625e-01, 0.000000000000000e+00, 0.000000000000000e+00, 7.580288899347621e-01, 0.000000000000000e+00, 0.000000000000000e+00, 2.195251205570665e+00, 0.000000000000000e+00, 0.000000000000000e+00, 1.705437435827845e+00, 0.000000000000000e+00, 0.000000000000000e+00, 4.887337278265103e-02, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
