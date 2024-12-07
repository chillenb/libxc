
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_bhandh_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_bhandh", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-9.234612420002182e-01, -6.521790495791597e-01, -1.305568372951491e-01, -7.846335333986669e-02, -3.110930962920290e-02, -7.775769074628854e-03, -1.364287525690647e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_bhandh_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_bhandh", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.220787319373473e+00, -1.221661879372637e+00, -8.678448693704459e-01, -8.683600538373578e-01, -2.973213724224406e-01, -2.976639111367038e-01, -1.046554373003040e-01, -8.947919945900491e-02, -4.147908622561509e-02, -3.052941074045570e-02, -1.031161925691502e-02, -1.034724535454612e-02, -1.743878875219199e-04, -2.027975801107367e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_bhandh_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_bhandh", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [6.601419480515535e-06, 5.222815421851711e-06, 6.477346546289099e-06, 4.053330432132051e-05, 3.646941789248587e-05, 3.987902204897787e-05, 3.461076381698536e-02, 4.773762863586187e-02, 3.476915844521113e-02, -4.194706172367205e-04, 4.596134769453040e+00, 3.448547549012627e+00, 2.178425425261535e-06, 2.356939734329661e+01, 1.767704995343962e+01, 4.086933530334335e-02, 7.936097321777658e-02, 4.107927410229123e-02, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
