
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_lc_blyp_ea_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_lc_blyp_ea", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.685160065762292e+00, -1.162504682959943e+00, -1.857980126757395e-01, -4.898858079676391e-02, -4.799883852589865e-03, -2.110231377901774e-03, -3.003792852316161e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_lc_blyp_ea_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_lc_blyp_ea", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.154156322724285e+00, -2.156085915753329e+00, -1.448068037648373e+00, -1.449237579357275e+00, -3.589073510953432e-01, -3.592674008795687e-01, -8.341205920686651e-02, -8.228458477467483e-02, -9.176992532505715e-03, -3.029946605911367e-02, -2.752183834973020e-03, -2.841859827475232e-03, -2.079739193384301e-05, -9.360858405529434e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_lc_blyp_ea_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_lc_blyp_ea", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-2.284625742295500e-04, 5.222815421851711e-06, -2.278434439703480e-04, -7.808282406503002e-04, 3.646941789248587e-05, -7.790812769886567e-04, 2.824998430629867e-03, 4.773762863586187e-02, 3.030974424862790e-03, -5.545705838960927e-01, 4.596134769453040e+00, 3.448466363201434e+00, -3.631194165902077e-01, 2.356939734329661e+01, 1.767704993544899e+01, 4.077523370589584e-02, 7.936097321777658e-02, 4.099121473553190e-02, -4.477220874981180e-09, 0.000000000000000e+00, -2.165913737296630e-09])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
