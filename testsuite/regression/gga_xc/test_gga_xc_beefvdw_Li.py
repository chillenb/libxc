
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_xc_beefvdw_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_beefvdw", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.921489349045908e+00, -1.385868848323364e+00, -4.943843092749161e-01, -1.814874233214371e-01, -9.425249486804269e-02, -2.536711354935540e-02, -4.987984300943183e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_xc_beefvdw_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_beefvdw", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.423544351065041e+00, -2.425679524357728e+00, -1.633378732983249e+00, -1.634724042303516e+00, -4.968621530642748e-01, -4.975198356494251e-01, -2.352857713315068e-01, -1.407323291268218e-01, -7.455018004831290e-02, 9.328208250105793e-02, -3.359260064704464e-02, -3.344445524044226e-02, -6.931139798044773e-04, -5.825592173969931e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_xc_beefvdw_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_beefvdw", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-2.191268643699832e-04, 3.674858595806771e-05, -2.181533593307246e-04, -1.097912338762438e-03, 1.191901136155099e-04, -1.093827434714746e-03, -7.700478107841084e-02, 2.498938991756018e-03, -7.658887533940577e-02, -1.811868150535716e+00, 2.703781805776042e+00, 1.167565359226088e+00, -1.095529627711735e+02, 9.031035206560485e+00, 3.336262335670375e+00, -1.872429896105830e-01, 1.342310948210240e-04, -1.748491894130950e-01, -8.584539127734145e-01, 1.284619440510545e-06, -1.228788781448848e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
