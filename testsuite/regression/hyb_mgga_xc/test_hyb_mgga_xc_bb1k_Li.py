
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_xc_bb1k_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_bb1k", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.090594992926128e+00, -7.870089806463645e-01, -2.603846573369611e-01, -9.323777116416608e-02, -4.713221877123291e-02, -7.727256842426522e-02, -3.109724251406577e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_xc_bb1k_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_bb1k", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.363466441578268e+00, -1.364493982698056e+00, -9.450039901390206e-01, -9.456022826683298e-01, -2.257248905802217e-01, -2.255722584291885e-01, -1.208776665179640e-01, -2.157942659232678e-02, -4.365859438703600e-02, -4.481265754530775e-03, -2.137660607980697e-02, -2.138139704784788e-02, -4.335151073928539e-03, -3.742797287119594e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_bb1k_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_bb1k", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.175759909814642e-04, 0.000000000000000e+00, -1.172387916244061e-04, -4.382421896731071e-04, 0.000000000000000e+00, -4.371809305919185e-04, -5.680698776326668e-02, 0.000000000000000e+00, -5.686545811150685e-02, -5.518361508094083e-02, 0.000000000000000e+00, -7.713133744686176e+02, -3.790429472225554e+01, 0.000000000000000e+00, -2.813109590470000e+07, -6.746594535822554e+02, 0.000000000000000e+00, -6.766197400957878e+02, -8.350234026730312e+07, 0.000000000000000e+00, -2.487974491247112e+08]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_bb1k_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_bb1k", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_bb1k_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_bb1k", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-1.535420278481735e-03, -1.534572015769308e-03, -1.754667312298453e-03, -1.753913243294452e-03, -4.414736987621208e-04, -4.386980204772437e-04, -9.437213115668527e-02, -1.176431803685671e-07, -1.559459506086730e-02, -3.855053562272995e-11, -1.373116706943418e-07, -1.256638198788993e-07, -8.551510937470655e-12, -4.468127138692301e-12]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
