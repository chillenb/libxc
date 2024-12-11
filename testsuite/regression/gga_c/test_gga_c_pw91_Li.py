
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_pw91_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_pw91", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-6.285424634539820e-02, -4.708577829899061e-02, -3.990548841963676e-03, -1.521984741125060e-02, -1.451865117542804e-03, -7.852165179352000e-09, -1.786494129712990e-16])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_pw91_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_pw91", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.126939971374459e-01, -1.125662792109756e-01, -9.923133440616196e-02, -9.913373006298408e-02, -1.980350172654639e-02, -1.981062597571861e-02, -2.406123642484551e-02, -9.702582900766886e-02, -6.964394754637950e-03, 4.054860501258080e-01, -5.089023701511680e-08, -5.114538147276165e-08, -1.136115127756826e-15, -1.343712738733488e-15])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_pw91_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_pw91", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([3.983663070495298e-05, 7.967326140990596e-05, 3.983663070495298e-05, 1.307708120093601e-04, 2.615416240187202e-04, 1.307708120093601e-04, 3.790454806682013e-03, 7.580909613364026e-03, 3.790454806682013e-03, 3.260964096831356e+00, 6.521928193662712e+00, 3.260964096831356e+00, 1.182151125193423e+01, 2.364302250386845e+01, 1.182151125193423e+01, 1.732042923415531e-04, 3.464085846831062e-04, 1.732042923415531e-04, 1.606953340538462e-06, 3.213906681076925e-06, 1.606953340538462e-06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
