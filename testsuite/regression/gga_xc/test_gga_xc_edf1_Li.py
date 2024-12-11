
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_xc_edf1_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_edf1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.881607388452087e+00, -1.343870360255777e+00, -4.397590085985275e-01, -1.631178638462503e-01, -8.240388070206883e-02, -2.484130064056319e-01, -1.045566357748195e-01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_xc_edf1_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_edf1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.409562068465172e+00, -2.411606676267110e+00, -1.651153647010504e+00, -1.652422640962820e+00, -3.722050110565625e-01, -3.724188140278141e-01, -2.134008179530368e-01, -1.470202303105156e-01, -6.211139764861294e-02, -4.805433199119000e-02, -5.438757643960477e-02, -5.501955093614234e-02, -1.426519821607456e-02, -1.247548307159059e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_xc_edf1_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_edf1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.412053187030458e-04, 6.980348033383547e-06, -1.407715809180628e-04, -6.862287610020352e-04, 4.873632963476560e-05, -6.844508502639797e-04, -1.379937938632408e-01, 6.393759384675643e-02, -1.378581145445153e-01, -1.979532306440735e+00, 6.255874372616108e+00, -2.624078805213484e+03, -1.045109583953521e+02, 3.302754432769007e+01, -9.477490319855677e+07, -2.286479017808994e+03, 1.294317979282627e-01, -2.289832960498306e+03, -2.813618165517662e+08, 0.000000000000000e+00, -8.381321210883077e+08])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
