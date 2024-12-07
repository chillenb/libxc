
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
    ref_tgt = [-5.035479748759288e-02, -3.149130735780180e-02, -1.449166633535152e-02, -1.341260546498480e-02, 2.860707736668244e-09, -5.542602059633343e-04, -5.163732578039568e-07]
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
    ref_tgt = [-1.257487268362697e-01, -1.256706445617689e-01, -8.516431659480404e-02, -8.514710055904331e-02, -2.402966065391891e-02, -2.398624500989338e-02, -2.993900190510840e-02, 4.519859130067578e-01, -2.078916869595052e-08, "nan", 5.698368432631353e-04, "nan", -3.295632137258997e-07, "nan"]
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
    ref_tgt = [6.655993950415153e-05, 1.189320872354904e-04, 6.650331852457894e-05, 1.395325456777449e-04, 2.754224434009556e-04, 1.396056939591803e-04, -1.149415347419893e-02, 3.145729209076590e-02, -1.152891429721135e-02, 7.238101427619954e+00, 1.339073388155134e+01, -1.472714224219834e+04, -2.342004452370075e-05, 1.572900382226780e+02, "nan", -3.487510678418494e+01, 8.364973340206798e+01, "nan", 6.148007549004456e+02, 1.229793600325573e+03, "nan"]
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
    ref_tgt = [-7.786576523105760e-04, -7.782776615656650e-04, -6.846352106558475e-04, -6.848901696159385e-04, 0.000000000000000e+00, 0.000000000000000e+00, -3.916402097022764e-02, -6.672106897267084e-04, 0.000000000000000e+00, "nan", -4.484567107467169e-06, "nan", -4.138414643855236e-17, "nan"]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_scanl_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_scanl", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
