
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_x_rel_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_x_rel", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.732890248232275e+00, -1.204272316871665e+00, -2.893667467981168e-01, -1.568996674851774e-01, -6.221852007736686e-02, -1.139515997998508e-02, -2.127820881238722e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_x_rel_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_x_rel", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.307312682962631e+00, -2.309393585331051e+00, -1.604286446338191e+00, -1.605652541062129e+00, -3.858942714864189e-01, -3.857301572805341e-01, -2.092633230874031e-01, -1.449947257966022e-02, -8.295797781657593e-02, -4.598292111904698e-04, -1.524839027047696e-02, -1.513748750074323e-02, -3.071820507045573e-04, -2.183783727793184e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
