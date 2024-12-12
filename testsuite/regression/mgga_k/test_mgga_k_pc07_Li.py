
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_k_pc07_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_pc07", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([4.217372862378161e+00, 4.099882256383768e+00, 3.507339944885866e+00, 2.114639622536349e-02, 7.165935031675759e-02, 3.195073348456298e+00, 1.356911237143376e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_k_pc07_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_pc07", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-4.217419202003022e+00, -4.217326647787665e+00, -4.098796785522157e+00, -4.100964960284972e+00, 3.869663458604229e-01, 3.744911381639358e-01, -2.007652420859430e-02, -3.423534008413791e+00, -7.165915626259604e-02, -1.207560080843476e+00, -3.030326141786138e+00, -3.621019360257141e+00, -1.418446622223791e+00, -1.185312171547316e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_k_pc07_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_pc07", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([1.927193223082906e-02, 0.000000000000000e+00, 1.921993237398214e-02, 5.758341797473614e-02, 0.000000000000000e+00, 5.743663279313743e-02, 4.106227717266546e-01, 0.000000000000000e+00, 4.282953981563652e-01, 2.605032494947879e+01, 0.000000000000000e+00, 9.520576043677614e+04, 4.181517806762772e+02, 0.000000000000000e+00, 2.450484496004929e+09, 6.736020890303675e+04, 0.000000000000000e+00, 8.731844401560794e+04, 8.236143115961227e+09, 0.000000000000000e+00, 2.291994335955933e+10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_k_pc07_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_pc07", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = numpy.asarray([0.000000000000000e+00, 0.000000000000000e+00, 1.101966523321404e-39, 8.567651879246898e-38, 1.713294143554540e-01, 1.704209067691722e-01, 0.000000000000000e+00, -3.014343000988006e-02, 0.000000000000000e+00, 4.346301649025606e-04, -1.904315700607978e-06, -3.874055423841186e-02, -2.315737076769605e-16, 5.146587533288480e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
