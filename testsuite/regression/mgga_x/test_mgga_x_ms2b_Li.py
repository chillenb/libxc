
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_ms2b_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ms2b", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.837710796792943e+00, -1.217444633236013e+00, -2.835926880022138e-01, -1.705084517548438e-01, -6.145894354189387e-02, -1.713407509305537e-02, -2.969797531463519e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_ms2b_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ms2b", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.653987175417436e+00, -2.656767924351342e+00, -1.715066923641149e+00, -1.717295853311876e+00, -3.818932789631774e-01, -3.818819377008707e-01, -2.386938896402503e-01, -2.177864853405507e-02, -8.330671859766929e-02, -6.916765007599792e-04, -2.293285346851217e-02, -2.273321948180553e-02, -4.620018042946411e-04, -2.121956352591760e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ms2b_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ms2b", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-8.369272196862112e-04, 0.000000000000000e+00, -8.343145567555671e-04, -1.847061377475407e-03, 0.000000000000000e+00, -1.849986558865461e-03, -2.400978029101659e-01, 0.000000000000000e+00, -2.409413330785579e-01, -1.134257980008597e+01, 0.000000000000000e+00, -1.938331164652770e-01, -1.201730405288748e+02, 0.000000000000000e+00, -1.241225136352803e+00, -8.267744164721487e-05, 0.000000000000000e+00, -1.839282898563473e-01, -5.665424900862919e-11, 0.000000000000000e+00, -2.166029254915013e+12])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ms2b_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ms2b", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([2.922374569065244e-02, 2.921744714217566e-02, 1.353607569626722e-02, 1.364373382689528e-02, 6.768835553218666e-04, 7.083498410655520e-04, 3.209350571630216e-01, 1.618681424715014e-24, 1.139935079194836e-02, 1.517387124930945e-30, -1.376088501969800e-30, 6.500920864628288e-25, -2.484473852571390e-41, 1.061028777096254e-12])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
