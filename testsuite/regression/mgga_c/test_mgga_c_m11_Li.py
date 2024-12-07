
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_m11_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m11", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.346118146568628e-01, -1.205877352081610e-01, 6.275225094816090e-03, -1.605281039862350e-02, 3.069923655185842e-03, -5.774974462847238e-02, -1.433180880273537e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_m11_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m11", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [1.920318638769734e-01, 1.922038032422150e-01, -3.851032719277881e-01, -3.849412230567547e-01, 5.725762740434533e-02, 5.725344346024910e-02, -5.678942524392851e-03, -1.358281570228952e-01, 1.297791720859465e-02, 1.022848742018111e+00, -7.256960917795860e-02, -7.338415977734236e-02, -1.685989816226630e-03, -2.473953846939804e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_m11_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m11", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.281552433137418e-04, -2.563104866274836e-04, -1.281552433137418e-04, -2.503201579226492e-04, -5.006403158452984e-04, -2.503201579226492e-04, 1.024584018924733e-02, 2.049168037849467e-02, 1.024584018924733e-02, -9.871758701348075e+00, -1.974351740269615e+01, -9.871758701348075e+00, 3.146533543141018e+01, 6.293067086282034e+01, 3.146533543141018e+01, 1.900754399990124e-03, 3.801508800066035e-03, 1.900754399990124e-03, 1.819491125847186e-05, 3.638988016866870e-05, 1.819491125847186e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_m11_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m11", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_m11_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m11", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-2.844511941139935e-02, -2.844511941139929e-02, 4.994363384322870e-02, 4.994363384322870e-02, -1.669932715395717e-02, -1.669932715395668e-02, 1.709214820081464e-01, 1.709214820081086e-01, -1.938659235575469e-01, -1.938659234239102e-01, -2.372866751829901e-07, -2.372866751902356e-07, -6.268023767883432e-19, -6.268605373074586e-19]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
