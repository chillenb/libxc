
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_dk87_r2_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_dk87_r2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-6.218444997183885e-01, -5.788181496140590e-01, -3.625772064548302e-01, -1.387335518821520e-01, -6.016097260210383e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_dk87_r2_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_dk87_r2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-8.284601919683300e-01, -9.290074480426683e-17, -7.115739629714141e-01, -1.961703624912082e-16, -4.028010436105334e-01, -1.464051111571296e-17, -1.144142697499469e-01, -7.638307419931547e-17, -7.132188505213896e-03, -3.519534307761746e-18]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_dk87_r2_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_dk87_r2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.021262203490427e-02, 0.000000000000000e+00, 0.000000000000000e+00, -2.795466164260759e-02, 0.000000000000000e+00, 0.000000000000000e+00, -1.761019903754711e-01, 0.000000000000000e+00, 0.000000000000000e+00, -7.942657980856413e+00, 0.000000000000000e+00, 0.000000000000000e+00, -5.839858886672218e+04, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
