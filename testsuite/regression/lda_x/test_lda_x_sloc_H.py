
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_x_sloc_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_x_sloc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.100020542187781e+00, -9.971484325492811e-01, -6.153619071767833e-01, -1.875317282295331e-01, -1.199170174336395e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_x_sloc_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_x_sloc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.430026704844120e+00, -6.487891719864081e-05, -1.296292962314072e+00, -6.482472173030551e-05, -7.999704793298369e-01, -6.497419621820377e-05, -2.437912466986906e-01, -6.501647875417627e-05, -1.558921244765182e-02, -6.501678201976252e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
