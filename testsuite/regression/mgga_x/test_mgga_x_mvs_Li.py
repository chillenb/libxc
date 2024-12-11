
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_mvs_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mvs", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.972016167764995e+00, -1.333400630174776e+00, -2.574539428551704e-01, -1.802761441416139e-01, -5.645194542053007e-02, -2.025550296220175e-03, -3.819657367863274e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_mvs_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mvs", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.719983548522730e+00, -2.722521426955848e+00, -1.903610022013522e+00, -1.904688721508972e+00, -3.510321640946260e-01, -3.512190045582838e-01, -2.462434036691937e-01, 2.012228394931681e+00, -7.871001517044494e-02, 5.654657574269105e+00, 3.616267107007822e+01, 2.008058189747759e+00, 5.220979863117859e+04, -1.906393239602881e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mvs_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mvs", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.522651512619254e-04, 0.000000000000000e+00, -2.514313277529567e-04, -1.065960351201267e-03, 0.000000000000000e+00, -1.058322699897707e-03, -5.858004800565570e-03, 0.000000000000000e+00, -6.154537106544259e-03, -4.295859420600966e+00, 0.000000000000000e+00, -5.166182589867621e+04, -1.204662823877523e+01, 0.000000000000000e+00, -1.146788143810724e+10, -1.642254681535280e+04, 0.000000000000000e+00, -4.415602585859202e+04, -2.400478930523685e+09, 0.000000000000000e+00, 9.701219334414449e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mvs_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mvs", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [1.308986727903392e-02, 1.308180085447736e-02, 1.851158525652194e-02, 1.842591824122663e-02, 1.410409616521344e-03, 1.479914871155827e-03, 1.649061740662438e-01, 6.600360450710112e-01, 2.880922381651038e-02, 4.672425105447126e+00, 2.438959976223611e-01, 6.417982909407548e-01, 2.914566802705019e-01, 2.089904399413017e-12]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
