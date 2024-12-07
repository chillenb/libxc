
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_x_rae_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_x_rae", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.064414213916571e-01, -7.389965589678735e-02, -1.774173512837072e-02, -9.619467641586929e-03, -3.814566167392496e-03, -6.986268584350001e-04, -1.304547457106965e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_x_rae_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_x_rae", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.418578999561980e-01, -1.419857177485238e-01, -9.849090612912052e-02, -9.857473594784769e-02, -2.366067473587064e-02, -2.365061251965600e-02, -1.282992835696371e-02, -8.890199551845874e-04, -5.086089084902371e-03, -2.819561280903820e-05, -9.348649285146815e-04, -9.280655809191316e-04, -1.883304965351525e-05, -1.338857764740943e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
