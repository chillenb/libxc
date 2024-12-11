
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_llp_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_llp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([1.645903672229089e+01, 8.178245862174112e+00, 6.532868196616374e-01, 1.325733365724945e-01, 2.647155222794557e-02, 7.808444671309594e-03, 5.896049097668570e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_llp_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_llp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([2.597929177806884e+01, 2.602669136170214e+01, 1.240633413468594e+01, 1.242767105461138e+01, 7.373733451210873e-01, 7.369435386259778e-01, 2.137001327625254e-01, 4.472494124795194e-03, 3.292775055936518e-02, 4.534862441642774e-05, 4.772524071858198e-03, 4.779142260536500e-03, 3.059232133447653e-05, 1.922117414177669e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_llp_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_llp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([2.447900304934207e-03, 0.000000000000000e+00, 2.442004396443883e-03, 6.389998510011336e-03, 0.000000000000000e+00, 6.375417141011260e-03, 1.704901323247489e-01, 0.000000000000000e+00, 1.703670806003500e-01, 3.564145355202469e+00, 0.000000000000000e+00, 7.477811656162721e+01, 2.448963092146617e+01, 0.000000000000000e+00, 8.587141703731718e+04, 6.837792017198340e+01, 0.000000000000000e+00, 6.798895862568192e+01, 1.702872360761933e+05, 0.000000000000000e+00, 3.606207238064662e+05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
