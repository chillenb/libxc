
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_m08_hx_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m08_hx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.826250829114105e-02, -1.182214707247656e-01, -1.527318571937984e-02, -1.722365679176378e-02, -7.014688485822915e-04, -4.716335465747555e-02, -1.170169207487549e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_m08_hx_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m08_hx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [9.658085368113859e-01, 9.659211492241112e-01, -4.638768292976293e-01, -4.637108952660255e-01, 1.035026722003372e-01, 1.034751637474024e-01, 1.161848717574814e-02, -4.451374859505308e-02, 6.609284889445471e-03, 1.542955854356409e+00, -5.928492902658812e-02, -5.995015543795982e-02, -1.376583649924958e-03, -2.019943645884915e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_m08_hx_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m08_hx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [1.696361692194340e-04, 3.392723384388680e-04, 1.696361692194340e-04, -1.985649799297842e-04, -3.971299598595683e-04, -1.985649799297842e-04, 1.491031798152344e-02, 2.982063596304689e-02, 1.491031798152344e-02, 1.083910030507204e+01, 2.167820061014408e+01, 1.083910030507204e+01, 4.877611806021581e+01, 9.755223612043157e+01, 4.877611806021581e+01, -4.329395135513153e-03, -8.658790271221704e-03, -4.329395135513153e-03, -4.145641938135391e-05, -8.291297011998979e-05, -4.145641938135391e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_m08_hx_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m08_hx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_m08_hx_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m08_hx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-1.665499527736952e-01, -1.665499527736950e-01, 5.913438420914015e-02, 5.913438420914013e-02, -3.314099489035634e-02, -3.314099489035834e-02, -1.433246975496919e+00, -1.433246975496605e+00, -2.420862941421227e-01, -2.420862939752478e-01, -3.848121686897536e-08, -3.848121672986291e-08, -9.657231084892361e-20, -9.667570732735109e-20]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
