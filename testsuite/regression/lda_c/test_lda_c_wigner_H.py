
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_wigner_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_wigner", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([3.841189307007511e-18, 3.698451259238471e-18, -2.858989782429729e-18, -1.200340366494216e-18, -8.444169699518191e-20])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_wigner_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_wigner", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([1.732499763926758e-17, -2.016355112088658e-01, 2.285386144092982e-17, -1.991928019750023e-01, -3.035304106148260e-18, -1.838950791354987e-01, -1.384181008042924e-18, -1.219654547659677e-01, -1.056113189324177e-18, -1.184869132955552e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
