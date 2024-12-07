
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_b88_6311g_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_b88_6311g", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.814527227624112e+00, -1.304959395854140e+00, -4.433742134121196e-01, -1.612902298644427e-01, -8.322269386363011e-02, -1.335158913062690e-01, -5.361652579338440e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_b88_6311g_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_b88_6311g", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.233030246453306e+00, -2.235148087468857e+00, -1.517566155790730e+00, -1.518919339777435e+00, -3.482824856431106e-01, -3.481797569784610e-01, -2.044186896800249e-01, -3.631550709735569e-02, -7.414890229291679e-02, -7.731589349371787e-03, -3.719363372435193e-02, -3.736233222819947e-02, -7.465651169509579e-03, -6.456076187472217e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_b88_6311g_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_b88_6311g", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-3.175868763319170e-04, 0.000000000000000e+00, -3.165526669706799e-04, -1.168006435521070e-03, 0.000000000000000e+00, -1.164398061373759e-03, -1.177618723253108e-01, 0.000000000000000e+00, -1.177146102766005e-01, -5.181146809908687e+00, 0.000000000000000e+00, -1.338585398379994e+03, -8.055912667238240e+01, 0.000000000000000e+00, -4.850208548103876e+07, -1.163935774385508e+03, 0.000000000000000e+00, -1.165819592262211e+03, -1.439986532617938e+08, 0.000000000000000e+00, -4.289576638328403e+08]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
