
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_k_rda_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_rda", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-8.840836051179392e+01, 6.893363934471852e+00, 2.151714642451387e+00, 9.844525321376350e-02, 5.427517328650134e-02, 3.077794931336344e+00, 1.356894989465323e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_k_rda_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_rda", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-8.980804052511448e+01, -8.980008020242236e+01, 9.984036206358500e+00, 9.998823091254014e+00, -1.122705546819759e+00, -1.173605031251195e+00, 2.771216570435502e-01, -3.066876306136494e+00, 2.839197421786298e-02, -1.210237631866150e+00, -3.043395752918351e+00, -3.148697614525422e+00, -1.418451651952786e+00, -1.185588894297124e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_k_rda_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_rda", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([1.594816039531064e-02, 0.000000000000000e+00, 1.589784793095780e-02, 5.618527961848789e-03, 0.000000000000000e+00, 5.596105229437263e-03, 1.707671064959920e+00, 0.000000000000000e+00, 1.754038195116651e+00, 1.541089245106952e+00, 0.000000000000000e+00, 7.829261099728307e+04, 1.041883065925668e+02, 0.000000000000000e+00, 2.454376727049230e+09, 6.733474705820995e+04, 0.000000000000000e+00, 6.882135102789881e+04, 8.236143114997760e+09, 0.000000000000000e+00, 2.292358375074679e+10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_k_rda_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_rda", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = numpy.asarray([2.448791899756971e-02, 2.454626967179398e-02, -1.072351780577941e-02, -1.082902680387133e-02, 4.402353004525607e-02, 4.237510723952123e-02, 9.875380806142699e-02, 7.884781202268341e-06, 3.447139406119509e-02, 4.754394887867289e-12, 7.405783522751487e-10, 6.862802846556976e-06, 7.769769804467622e-23, 1.829274099154157e-13])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
