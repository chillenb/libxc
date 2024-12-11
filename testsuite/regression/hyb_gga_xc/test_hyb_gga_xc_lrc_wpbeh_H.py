
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_lrc_wpbeh_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_lrc_wpbeh", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-4.452338041796057e-01, -3.941417040264885e-01, -2.107571144768765e-01, -3.504760527180138e-02, -2.695577024339153e-06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_lrc_wpbeh_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_lrc_wpbeh", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-6.133402086753788e-01, 1.596983044605467e+00, -5.325701181989345e-01, 7.387469275550377e+01, -2.694691579122566e-01, 4.139501778593020e+01, -3.929652266005980e-02, 3.382974671448072e-01, -5.398272972252937e-06, 1.066995416409889e-08])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_lrc_wpbeh_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_lrc_wpbeh", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([5.268470252902259e-03, 3.288056552649452e-02, 1.644028276324726e-02, -6.785983540543116e-03, 2.041382516785488e-02, 1.020691258392744e-02, -7.520199526538357e-02, 8.417394271524091e-02, 4.208697135762044e-02, -2.544192236755769e+00, 1.762212799391784e-01, 8.811063996958873e-02, 9.608925756618611e-04, 1.985451945004567e-03, 9.927259729551347e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
