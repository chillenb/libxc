
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_x_s12h_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_x_s12h", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.346512492422983e+00, -9.611737359997471e-01, -3.198208144484959e-01, -1.209537180723984e-01, -6.235350161486533e-02, -1.500969236547180e-02, -2.803935007493714e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_x_s12h_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_x_s12h", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.725181059327372e+00, -1.726941526638252e+00, -1.113654386994746e+00, -1.114771145685140e+00, -3.235305394085906e-01, -3.236941453400408e-01, -1.594991261279731e-01, -1.908618394792859e-02, -5.927476153173816e-02, -6.060221162957373e-04, -2.006776523095626e-02, -1.992303126464716e-02, -4.047886956381068e-04, -2.877679065799458e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_x_s12h_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_x_s12h", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.189084962364502e-04, 0.000000000000000e+00, -1.182370041669404e-04, -8.819901427711012e-04, 0.000000000000000e+00, -8.786225662903960e-04, -4.990789911670179e-02, 0.000000000000000e+00, -4.975120315019390e-02, -8.854812060145660e-01, 0.000000000000000e+00, -1.409102253271227e-01, -5.221835714336770e+01, 0.000000000000000e+00, -9.007496313932245e-01, -1.432058176599934e-01, 0.000000000000000e+00, -1.337242143572341e-01, -6.557118308626066e-01, 0.000000000000000e+00, -9.385837827131217e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
