
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_xc_oblyp_d_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_oblyp_d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.857372514702911e+00, -1.338985015212682e+00, -4.115363715656112e-01, -1.604488759736686e-01, -8.079691330234709e-02, -1.351502026599361e-01, -5.364294971644318e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_xc_oblyp_d_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_oblyp_d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.311220835432309e+00, -2.313174326749992e+00, -1.594202085055612e+00, -1.595394980743255e+00, -4.515603038684133e-01, -4.519193769536057e-01, -2.052458265716921e-01, -1.235043289092133e-01, -7.320535469430307e-02, -3.797666595103896e-02, -3.910316019659347e-02, -3.937044228580703e-02, -7.480571932300144e-03, -6.544660542566738e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_xc_oblyp_d_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_oblyp_d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-2.570211341418590e-04, 5.588321563163314e-06, -2.562806308907760e-04, -9.546407313148976e-04, 3.897657416167142e-05, -9.522262557859213e-04, -7.518738861049898e-02, 5.221555243047198e-02, -7.497236675097917e-02, -4.240280583248348e+00, 5.947707043493950e+00, -1.335392427660832e+03, -7.554626994732729e+01, 4.108238973103639e+01, -4.850325644486766e+07, -1.164778190563284e+03, 6.469362024933486e-01, -1.166628498421389e+03, -1.440006451225896e+08, 0.000000000000000e+00, -4.289620506838993e+08])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
