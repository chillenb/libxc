
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_k_l06_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_l06", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [1.634784894093005e+01, 8.129890608498819e+00, 8.881365672396141e-01, 1.318883369512776e-01, "nan", 1.535164330011899e-03, 5.454746473375978e-07]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_k_l06_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_l06", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [2.590277547844456e+01, 2.594942102420822e+01, 1.217943268069518e+01, 1.220095678747298e+01, 9.705376366665301e-01, 9.739417498467892e-01, 2.137405070357412e-01, 2.330518074326372e-03, "nan", "nan", 2.577075198856814e-03, "nan", 1.045854266538853e-06, "nan"]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_k_l06_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_l06", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [2.262217723326419e-03, 0.000000000000000e+00, 2.257756878069444e-03, 7.161447720826212e-03, 0.000000000000000e+00, 7.142226810215973e-03, 2.472423366899487e-01, 0.000000000000000e+00, 2.451520563398443e-01, 2.990408377248952e+00, 0.000000000000000e+00, 1.621348474501124e-05, 7.808523507794926e+01, 0.000000000000000e+00, "nan", 1.932235413168964e-05, 0.000000000000000e+00, "nan", 1.542777116817871e-09, 0.000000000000000e+00, "nan"]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_k_l06_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_l06", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [-1.308846636467413e-03, -1.327943184215802e-03, -2.686837643030157e-03, -2.683670159154295e-03, -4.261571104073102e-03, -4.220468472878439e-03, -7.741425739301749e-04, -1.634062101925836e-11, -1.273336846087146e-02, "nan", -2.264355851372527e-11, 0.000000000000000e+00, -1.478746386192364e-20, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_k_l06_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_l06", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
