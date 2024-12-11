
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_camy_pbeh_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_camy_pbeh", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-2.214833617010453e-01, -2.092122220210274e-01, -1.764908881719303e-01, -1.005514935486195e-01, -5.916414356256281e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_camy_pbeh_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_camy_pbeh", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.446419071728687e-01, 1.596983044605467e+00, -2.305856455568507e-01, 7.387469275550377e+01, -1.755137179705543e-01, 4.139501778593020e+01, -9.867848835854343e-02, 3.382974671448072e-01, -7.882425451770423e-03, 1.066995416435917e-08])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_camy_pbeh_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_camy_pbeh", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([1.107107981978541e-02, 3.288056552649452e-02, 1.644028276324726e-02, 1.594019899739795e-04, 2.041382516785488e-02, 1.020691258392744e-02, -5.930198589940196e-02, 8.417394271524091e-02, 4.208697135762044e-02, -3.520426965697503e+00, 1.762212799391784e-01, 8.811063996958873e-02, -4.428074024127073e+00, 1.985451945004567e-03, 9.927259729551347e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
