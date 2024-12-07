
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_pk09_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_pk09", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-3.251708508266784e-02, -3.118007569860862e-02, -2.480686129326408e-02, -1.332277067010814e-02, -1.628860747026377e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_pk09_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_pk09", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-3.663846986258387e-02, -2.928489970854706e+07, -3.523080587441239e-02, -2.091041146434232e+07, -2.868167251153693e-02, -4.091351715519316e+06, -1.566405611973359e-02, -8.137545883222122e+04, -2.064098842499560e-03, -8.527554157248101e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
