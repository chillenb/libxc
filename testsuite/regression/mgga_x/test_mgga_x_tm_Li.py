
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_tm_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_tm", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.749408291250698e+00, -1.209091538850342e+00, -3.514369708800404e-01, -1.584475419556881e-01, -6.642888757717044e-02, -6.558799535747394e-02, -2.689837550186930e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_tm_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_tm", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.346808606118339e+00, -2.348922006773355e+00, -1.625600852022150e+00, -1.627016112043685e+00, -3.532219893973077e-01, -3.518663459129239e-01, -2.122687894969246e-01, -2.085620783475742e-02, -7.370627178336267e-02, -2.445477128912461e-03, -5.244592009531068e-02, -2.155148277148766e-02, -2.073793546424148e-02, -3.277364701408872e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_tm_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_tm", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-4.158403978693418e-04, 0.000000000000000e+00, -4.143984146900499e-04, -1.037831491871724e-03, 0.000000000000000e+00, -1.036171436694317e-03, -5.037437968239022e-03, 0.000000000000000e+00, -5.631561183842360e-03, -9.454725737040635e+00, 0.000000000000000e+00, -5.047770576772503e+02, -2.201306149763553e+01, 0.000000000000000e+00, -3.497866012664654e+06, -1.673191940199520e+01, 0.000000000000000e+00, -4.456805425096432e+02, -6.734408188451059e+02, 0.000000000000000e+00, 1.408189365286133e+06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_tm_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_tm", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [2.216361831033069e-03, 2.190914954835255e-03, 2.072935860310701e-03, 2.076831130562991e-03, -2.118047097093876e-02, -2.151530734003178e-02, 2.711039958904221e-02, 3.895036744553971e-03, -1.244691425508199e-01, 7.367690545649392e-04, 1.285571263862510e-04, 3.912939835595237e-03, 4.225969770970404e-08, -5.528680090207400e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
