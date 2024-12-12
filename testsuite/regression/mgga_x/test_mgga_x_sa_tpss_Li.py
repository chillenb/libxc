
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_sa_tpss_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_sa_tpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.968904993322365e+00, -1.367167341769567e+00, -3.985811865419212e-01, -1.768320393796045e-01, -7.664960594904754e-02, -5.597954254812800e-01, -1.937128669895020e+01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_sa_tpss_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_sa_tpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.401408132812747e+00, -2.403567691499475e+00, -1.698618127651007e+00, -1.700289649464569e+00, -3.282407150120583e-01, -1.392959302393892e-01, -2.135661841993495e-01, 2.388887961171044e+01, -6.676482841873028e-02, 1.887070917516681e+02, -2.927272621947328e-01, 2.348414023466793e+01, -5.529382918598728e+00, -1.263713529264402e+02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_sa_tpss_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_sa_tpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.016193422154883e-03, 0.000000000000000e+00, -1.015321097362019e-03, -1.691204168642455e-03, 0.000000000000000e+00, -1.689103935659848e-03, -1.235983274555480e-01, 0.000000000000000e+00, -3.633148530742979e-01, -2.834784091382898e+01, 0.000000000000000e+00, -6.125154175034319e+05, -1.092391723171508e+02, 0.000000000000000e+00, -3.827052079962767e+11, -3.686152632849260e+03, 0.000000000000000e+00, -5.156818336986079e+05, -2.613171972347397e+10, 0.000000000000000e+00, 2.443411523929734e+12])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_sa_tpss_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_sa_tpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([5.268001004536731e-02, 5.281114051177847e-02, 2.889057157978863e-02, 2.898735841818023e-02, 9.728934924089577e-03, 6.635897381671640e-02, 1.023596663755798e+00, 7.822859259119674e+00, 1.212951812401963e-01, 1.559276550530430e+02, -2.891353944351656e-03, 7.492586586635507e+00, -5.887463585861491e-05, -1.065894212045638e+02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
