
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_lda_xc_cam_lda0_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_lda_xc_cam_lda0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.350398436851434e+00, -9.433437593734390e-01, -2.302843973684129e-01, -1.072087705447362e-01, -4.303589836412576e-02, -1.248270290376547e-02, -2.745649059674258e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_lda_xc_cam_lda0_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_lda_xc_cam_lda0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.792204914612845e+00, -1.793605532042352e+00, -1.252130450091165e+00, -1.253015750802512e+00, -3.063657438538632e-01, -3.062892852421298e-01, -1.443728669488479e-01, -1.315728513909969e-01, -5.646425004071365e-02, -7.175736789448879e-02, -1.615896663645899e-02, -1.619884451679207e-02, -3.514303218989595e-04, -3.994905177532429e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
