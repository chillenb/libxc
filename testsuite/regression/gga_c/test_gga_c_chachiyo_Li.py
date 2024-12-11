
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_chachiyo_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_chachiyo", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-5.753789906143260e-02, -4.206970683290507e-02, -4.931039092822963e-03, -1.490935870944068e-02, -8.662008844566748e-04, -1.764906853236746e-23, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_chachiyo_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_chachiyo", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.125227327648934e-01, -1.123779469497194e-01, -9.445911753076741e-02, -9.434628505638935e-02, -2.010039002804476e-02, -2.011289024543402e-02, -2.482512301142205e-02, -4.023659722196335e-01, -5.671322837540457e-03, -7.840960641509701e-01, -6.682239640224233e-22, -6.774078112825277e-22, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_chachiyo_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_chachiyo", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([4.538047455108476e-05, 9.076094910216953e-05, 4.538047455108476e-05, 1.344516594707104e-04, 2.689033189414208e-04, 1.344516594707104e-04, 3.564267248751470e-03, 7.128534497502940e-03, 3.564267248751470e-03, 3.981804830609263e+00, 7.963609661218527e+00, 3.981804830609263e+00, 1.052639318708615e+01, 2.105278637417230e+01, 1.052639318708615e+01, 1.972816231049214e-18, 3.945632462098427e-18, 1.972816231049214e-18, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
