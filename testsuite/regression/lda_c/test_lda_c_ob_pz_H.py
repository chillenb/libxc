
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_ob_pz_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_ob_pz", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-2.976228689275991e-02, -2.826995303005860e-02, -2.337749848463556e-02, -1.286530546084790e-02, -1.619481370113593e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_ob_pz_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_ob_pz", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-3.495830420831081e-02, -2.198041234558033e-01, -3.135512812104541e-02, -2.097509714557984e-01, -2.635232849012627e-02, -1.753579930387260e-01, -1.511254568706038e-02, -8.916844172720742e-02, -2.058108210344248e-03, -6.099695728849244e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
