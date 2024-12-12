
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_scanl_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_scanl", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-2.565339835151372e-02, -2.438245026061145e-02, -1.653481522274538e-02, -1.251244709499152e-04, -3.593221726584661e-08, -1.066114590787740e-03, -5.788721160931869e-06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_scanl_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_scanl", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.821749384963761e-02, -2.814484551968669e-02, -2.858443263516080e-02, -2.851419181177217e-02, -8.137014048840296e-02, -7.922760285893071e-02, -2.822350715031919e-06, 1.125590761835537e-01, -2.675475447321982e-08, 1.331356049988771e-01, -2.003922202131319e-03, -2.010317852588984e-03, -1.074288932825480e-05, -1.349324009730586e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_scanl_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_scanl", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.329689089480297e-05, 3.367080824267742e-05, -1.321558742151646e-05, -5.537822262796116e-05, 1.486126015899495e-04, -5.504764540641575e-05, 3.739876454874218e-02, 6.193323025887384e-02, 3.473726057478076e-02, 1.069276610905516e-02, 4.159233090384069e+00, -7.559187356336263e+03, 3.460549434711313e-05, 1.572923195650363e+02, -4.608878556476311e+08, 2.524273474644989e+00, 6.919367297480260e+00, 2.247119699659585e+00, 6.184326313190482e+03, 1.455167636336293e+04, 4.238325373913802e+03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_scanl_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_scanl", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = numpy.asarray([0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, -2.865771362019189e-03, -2.741715012224643e-03, 0.000000000000000e+00, 2.393999278377990e-03, 0.000000000000000e+00, -8.174538513111057e-05, 2.644463712400265e-11, 5.379779720822907e-07, 3.866407145159237e-23, -6.820621370419633e-12])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
