
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_k_csk_loc1_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_csk_loc1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [1.539265185657557e+01, 7.934478050880360e+00, 3.240711629275109e+00, 1.274177040757631e-01, "nan", 1.533263972368564e+166, 1.156734948703097e+258]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_k_csk_loc1_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_csk_loc1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [2.531546496661809e+01, 2.529772454173924e+01, 8.334796495794169e+00, 8.358866496750309e+00, -3.105207188593931e+00, -3.115263315167536e+00, 2.179754403185143e-01, -3.057065390161803e+00, "nan", "nan", -7.411761340967504e+150, -4.260319780195023e+150, 1.079010073129237e+243, 2.137755761021083e+243]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_k_csk_loc1_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_csk_loc1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [4.613283424226728e-04, 0.000000000000000e+00, 5.691199518373369e-04, 2.570866432673944e-02, 0.000000000000000e+00, 2.560256066096874e-02, 4.122313743354389e+00, 0.000000000000000e+00, 4.127806023374471e+00, -3.521452923393071e+00, 0.000000000000000e+00, 7.829811432416880e+04, 4.057988587616181e+02, 0.000000000000000e+00, "nan", 6.733474606521256e+04, 0.000000000000000e+00, -1.135621965072012e+04, 8.236143114997622e+09, 0.000000000000000e+00, -3.782391318881807e+09]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_k_csk_loc1_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_csk_loc1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [1.819120150036897e-01, 1.808547023027366e-01, 1.031652187298531e-01, 1.032968624728848e-01, 1.395217630046619e-03, 1.384977672207640e-03, 2.115671308393280e-01, 3.317085162717959e-09, 5.505790114797664e-03, "nan", 4.122957666927821e-09, 2.171250000000002e-01, 3.100410450014567e-15, 2.171249999999999e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_k_csk_loc1_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_csk_loc1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
