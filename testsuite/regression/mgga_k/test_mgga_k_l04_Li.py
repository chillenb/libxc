
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_k_l04_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_l04", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([2.623947522541970e+01, 8.236455844208765e+00, 7.520739138771941e-01, 1.345184431479706e-01, 2.986220139388777e-02, 1.233052841767768e-03, 4.381283454128215e-07])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_k_l04_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_l04", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([3.423405127936011e+01, 3.426045131653859e+01, 1.192876628065060e+01, 1.195028184239023e+01, 1.030944754757899e+00, 1.030483176744861e+00, 2.091846466304775e-01, 1.871886417572359e-03, 2.544782424572654e-02, 1.882867359881490e-06, 2.069923931185832e-03, 2.039923019454004e-03, 8.400361072296389e-07, 4.245467632258791e-07])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_k_l04_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_l04", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([6.432880536506701e-03, 0.000000000000000e+00, 6.440423893607846e-03, 8.159654621377327e-03, 0.000000000000000e+00, 8.137109341751554e-03, 1.041787677741559e-01, 0.000000000000000e+00, 1.047461166779557e-01, 3.673205030358899e+00, 0.000000000000000e+00, 4.284006063860679e-06, 5.537701754358345e+01, 0.000000000000000e+00, 3.653988317478039e-09, -2.607280190541929e-10, 0.000000000000000e+00, 4.792847995055850e-06, -1.252691200432097e-24, 0.000000000000000e+00, 4.148968039414005e-10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_k_l04_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_l04", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = numpy.asarray([-2.095032988289699e-03, -2.103574393969965e-03, -5.966693867718450e-03, -5.962043145994520e-03, 2.973008009968291e-04, 2.370928621269214e-04, -6.316296469200866e-03, 3.223794537855293e-12, -2.347355213819429e-03, -2.104829844536836e-20, 9.286451256558735e-16, 3.645017923631809e-12, 3.448992509092411e-35, -4.080820777397485e-22])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
