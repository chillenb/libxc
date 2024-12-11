
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_cam_qtp_02_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_cam_qtp_02", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.214915268209630e+00, -8.368370207166163e-01, -1.169630194458190e-01, -3.081189656201997e-02, -2.816250537925943e-03, -2.096691667271050e-03, -3.003783554145272e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_cam_qtp_02_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_cam_qtp_02", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.556394992050789e+00, -1.557735269158180e+00, -1.049159695663838e+00, -1.049952615294499e+00, -2.785551793173868e-01, -2.789294031568190e-01, -5.358058329273607e-02, -8.226105110928285e-02, -5.430567263984126e-03, -3.029946530784746e-02, -2.724821284191215e-03, -2.815089287183108e-03, -2.079716805595033e-05, -9.360850361890175e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_cam_qtp_02_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_cam_qtp_02", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.595579114426640e-04, 5.222815421851711e-06, -1.591596330973458e-04, -5.347321280479011e-04, 3.646941789248587e-05, -5.337197584498854e-04, 1.535964615207734e-02, 4.773762863586187e-02, 1.554997721345525e-02, -3.064747497568291e-01, 4.596134769453040e+00, 3.448509950965699e+00, -1.734380893036719e-01, 2.356939734329661e+01, 1.767704994510886e+01, 4.082575508618672e-02, 7.936097321777658e-02, 4.103849232540940e-02, -2.073226785857441e-09, 0.000000000000000e+00, -1.002950428704558e-09])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
