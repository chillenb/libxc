
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_hse06_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_hse06", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-5.134926257095780e-01, -4.654502650244288e-01, -2.939503338907336e-01, -1.152848716096177e-01, -7.396167115826132e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_hse06_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_hse06", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-6.734111085807046e-01, 1.596983044605467e+00, -5.918307313246619e-01, 7.387469275550377e+01, -3.394951757385128e-01, 4.139501778593020e+01, -1.210868315355013e-01, 3.382974671448072e-01, -9.859346322510345e-03, 1.066995416304534e-08]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_hse06_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_hse06", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [6.244806471435696e-03, 3.288056552649452e-02, 1.644028276324726e-02, -8.729564591409257e-03, 2.041382516785488e-02, 1.020691258392744e-02, -9.848185302010354e-02, 8.417394271524091e-02, 4.208697135762044e-02, -3.201755910329075e+00, 1.762212799391784e-01, 8.811063996958873e-02, 9.927259729551347e-04, 1.985451945004567e-03, 9.927259729551347e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
