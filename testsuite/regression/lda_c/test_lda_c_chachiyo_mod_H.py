
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_chachiyo_mod_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_chachiyo_mod", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-3.211960599988956e-02, -3.086576366918796e-02, -2.509093355661359e-02, -1.360858679307705e-02, -1.392621976151239e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_chachiyo_mod_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_chachiyo_mod", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-3.599076375032748e-02, -1.714907520909080e+03, -3.465656461012361e-02, -1.469261204057609e+03, -2.847719139860551e-02, -6.746758112571342e+02, -1.603021468315808e-02, -8.638839384275195e+01, -1.820585831613129e-03, -2.995183803597816e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
