
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_lc_wpbe_whs_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_lc_wpbe_whs", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-4.575601190656703e-01, -3.960326494450886e-01, -1.799899416246650e-01, -1.241099862362227e-02, -8.411113161868849e-07]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_lc_wpbe_whs_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_lc_wpbe_whs", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-6.594156889121380e-01, 1.596983044605467e+00, -5.645205678665508e-01, 7.387469275550377e+01, -2.570101893670229e-01, 4.139501778593020e+01, -1.705860981848095e-02, 3.382974671448072e-01, -1.683542342614995e-06, 1.066995416223924e-08]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_lc_wpbe_whs_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_lc_wpbe_whs", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [5.254786324018814e-03, 3.288056552649452e-02, 1.644028276324726e-02, -6.711675314283017e-03, 2.041382516785488e-02, 1.020691258392744e-02, -5.634296592842033e-02, 8.417394271524091e-02, 4.208697135762044e-02, -9.990731831507147e-01, 1.762212799391784e-01, 8.811063996958873e-02, 9.916139086909486e-04, 1.985451945004567e-03, 9.927259729551347e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
