
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_xc_b97m_v_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_b97m_v", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.215768791965622e+00, -1.550744085725783e+00, -3.163256303627867e-01, -8.442820538750141e-02, -9.570670421904892e-02, -2.614247816295617e-02, -7.054520468291065e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_xc_b97m_v_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_b97m_v", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.765450932891658e+00, -2.768144986123769e+00, -1.922556109674597e+00, -1.922847683113745e+00, -3.935313594002786e-01, -3.942272391830040e-01, -1.555969989836798e-01, 3.469634982146976e-01, -7.654336696453577e-02, 2.057428644581668e-01, -3.337120281786479e-02, -3.327935102712402e-02, -9.990038517490767e-04, -7.446534128287735e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_xc_b97m_v_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_b97m_v", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-4.340711692734288e-04, 0.000000000000000e+00, -4.308663083376312e-04, -2.550867820555476e-03, 0.000000000000000e+00, -2.544331590025722e-03, -7.009811918966918e-02, 0.000000000000000e+00, -7.165184471664426e-02, 7.691077376477870e+01, 0.000000000000000e+00, 1.096000083083132e+02, -7.090217265497637e+01, 0.000000000000000e+00, 8.132833026727632e+03, -4.785542768682511e-01, 0.000000000000000e+00, -4.077455769991175e-01, -3.093123857391631e+00, 0.000000000000000e+00, 2.917902732695372e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_xc_b97m_v_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_b97m_v", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_xc_b97m_v_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_b97m_v", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [4.572618283255363e-03, 4.480995959739750e-03, 4.930937065960105e-02, 4.916495268088671e-02, 2.576763056717489e-02, 2.656825739631088e-02, -3.912979852535024e+00, 1.027434924350854e-05, -1.069725099862643e-01, -1.921132224406661e-07, 2.267429268381026e-09, 4.717750062857038e-06, 3.178939873052388e-20, -1.313291195903799e-11]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
