
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_tfvw_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_tfvw", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([2.008347552464787e+01, 1.174763348014386e+01, 3.656566504668984e+00, 1.507078207676388e-01, 9.203630210542323e-02, 3.085155899215970e+00, 1.356897612707241e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_tfvw_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_tfvw", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([2.220223751458484e+01, 2.224996116352981e+01, 8.636597163952239e+00, 8.656117491217140e+00, -2.475998515816447e+00, -2.486210612314386e+00, 1.960303492379935e-01, -3.056027994469376e+00, -3.769756403480135e-02, -1.210226358618995e+00, -3.031000934736708e+00, -3.136824137922040e+00, -1.418446622334406e+00, -1.185586352382245e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_tfvw_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_tfvw", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([1.927193223082906e-02, 0.000000000000000e+00, 1.921993237398214e-02, 5.758341797473614e-02, 0.000000000000000e+00, 5.743663279313743e-02, 4.153406735139716e+00, 0.000000000000000e+00, 4.158710224823615e+00, 2.605032494947879e+01, 0.000000000000000e+00, 7.829811571772349e+04, 4.181517806762772e+02, 0.000000000000000e+00, 2.454376727176486e+09, 6.733474755479385e+04, 0.000000000000000e+00, 6.882557364072799e+04, 8.236143114997760e+09, 0.000000000000000e+00, 2.292358375079883e+10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
