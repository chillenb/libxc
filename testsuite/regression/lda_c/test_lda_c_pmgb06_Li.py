
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_pmgb06_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_pmgb06", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-3.343165365662783e-03, -4.425383472242190e-03, -1.171215745996499e-02, -1.158609637065600e-02, -1.058378984563277e-02, -6.746480991380847e-03, -1.681737084176587e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_pmgb06_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_pmgb06", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.475649233909628e-03, -2.472727201376260e-03, -3.310207059898622e-03, -3.306460344888140e-03, -9.573435646099099e-03, -9.571988852651236e-03, -1.018805596113682e-02, -1.128351083888516e-01, -1.222305557460858e-02, -7.547177634065269e-02, -8.458656543901634e-03, -8.551473972195803e-03, -1.978391109863612e-04, -2.903006865058161e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
