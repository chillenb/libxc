
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_mvs_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mvs", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.038062362557116e+00, -1.414595313739676e+00, -3.201273278128686e-01, -1.839820252091833e-01, -7.214899927015191e-02, -2.578952236048904e-03, -7.034023183992077e-06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_mvs_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mvs", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.636725553479810e+00, -2.639214055747735e+00, -1.775453477740454e+00, -1.776910445796521e+00, -2.401453917107332e-01, -1.479834563750842e-01, -2.415917319445477e-01, 2.012228394931681e+00, -6.311449429917963e-02, 5.201940707679225e+00, -4.455432226551041e-03, 2.008058189747759e+00, -1.540396492861856e-05, 1.528634075086902e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mvs_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mvs", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-3.656111167242909e-04, 0.000000000000000e+00, -3.643807866085098e-04, -1.565552120407670e-03, 0.000000000000000e+00, -1.562105480678659e-03, -3.026327101558025e-01, 0.000000000000000e+00, -4.212612702869213e-01, -5.403818377613428e+00, 0.000000000000000e+00, -5.166182589867621e+04, -2.095427575688911e+02, 0.000000000000000e+00, -1.054975859586553e+10, 1.236763317633570e+01, 0.000000000000000e+00, -4.415602585859202e+04, 1.118029540585464e+04, 0.000000000000000e+00, -2.955667042855389e+10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mvs_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mvs", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mvs_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mvs", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [1.904282594585162e-02, 1.902994622142505e-02, 2.739518817506581e-02, 2.740426440346975e-02, 8.043616538709220e-02, 1.090864385763467e-01, 2.078984202956744e-01, 6.600360450710112e-01, 5.247733728455336e-01, 4.298349314627122e+00, 1.057760031054564e-11, 6.417982909407548e-01, 9.756264250692077e-24, 1.289357535205589e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
