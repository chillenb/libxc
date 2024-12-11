
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_lc_qtp_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_lc_qtp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.590294770044676e+00, -1.067729752460373e+00, -1.120511976346119e-01, -2.604044613189604e-02, -2.011522907012964e-03, -2.090972717324664e-03, -3.003779626200086e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_lc_qtp_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_lc_qtp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.068313991659160e+00, -2.070229039923264e+00, -1.368904419470911e+00, -1.370059915644248e+00, -2.962461388413723e-01, -2.965787094957887e-01, -4.792635923987043e-02, -8.225111177340208e-02, -3.949055777206764e-03, -3.029946499048048e-02, -2.713265244440182e-03, -2.803783172586779e-03, -2.079707348035058e-05, -9.360846963912821e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_lc_qtp_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_lc_qtp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-2.075500927260294e-04, 5.222815421851711e-06, -2.070168461925779e-04, -6.730914973547948e-04, 3.646941789248587e-05, -6.717565108918327e-04, 2.136888895904120e-02, 4.773762863586187e-02, 2.155680172029809e-02, -1.587008913416322e-01, 4.596134769453040e+00, 3.448534626916530e+00, -6.354688633314649e-02, 2.356939734329661e+01, 1.767704995057705e+01, 4.085435668297135e-02, 7.936097321777658e-02, 4.106525743101442e-02, -7.123921297230947e-10, 0.000000000000000e+00, -3.446288624674222e-10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
