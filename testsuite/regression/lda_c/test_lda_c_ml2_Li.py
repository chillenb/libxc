
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_ml2_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_ml2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-3.929383086556310e-02, -3.743197192404392e-02, -2.530844284421658e-02, -2.504776701184473e-03, -8.151281097275820e-05, -2.431658429706257e-03, -4.401909911300383e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_ml2_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_ml2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-4.077709175315204e-02, -4.077758845046568e-02, -3.936182493254314e-02, -3.935953086315772e-02, -2.883919788867050e-02, -2.884590204428832e-02, -2.438384929846013e-03, -2.475426952367502e+00, -8.124602411402493e-05, -1.600583360648683e+02, -3.164102097828320e-03, -3.193752428920039e-03, -4.985475629581867e-05, -8.313737061615570e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
