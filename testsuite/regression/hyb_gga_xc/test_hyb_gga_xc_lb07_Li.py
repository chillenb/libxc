
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_lb07_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_lb07", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.378782769734866e+00, -9.016464599255523e-01, -7.602698240245094e-02, -2.151806930567649e-02, -1.624127735508418e-03, -2.088559670216724e-03, -3.003777978356361e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_lb07_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_lb07", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.902280846072310e+00, -1.903980460567305e+00, -1.270736578768717e+00, -1.271786627447902e+00, -2.568662317558597e-01, -2.571681832058464e-01, -4.006432175725236e-02, -8.224691306083619e-02, -3.205710843871107e-03, -3.029946485733866e-02, -2.708380068063723e-03, -2.799004154459044e-03, -2.079703380414646e-05, -9.360845538401072e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_lb07_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_lb07", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [6.601419480515535e-06, 5.222815421851711e-06, 6.477346546289099e-06, 4.053330432132051e-05, 3.646941789248587e-05, 3.987902204897787e-05, 3.461076381698536e-02, 4.773762863586187e-02, 3.476915844521113e-02, -4.194706172367205e-04, 4.596134769453040e+00, 3.448547549012627e+00, 2.178425425261535e-06, 2.356939734329661e+01, 1.767704995343962e+01, 4.086933530334335e-02, 7.936097321777658e-02, 4.107927410229123e-02, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
