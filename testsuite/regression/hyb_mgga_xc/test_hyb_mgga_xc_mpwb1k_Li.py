
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_xc_mpwb1k_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_mpwb1k", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.054943599711664e+00, -7.615833393759056e-01, -2.488145142187272e-01, -9.002606370188392e-02, -4.537352832871922e-02, -9.720410164926203e-04, -1.163729010154000e-06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_xc_mpwb1k_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_mpwb1k", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.318464478973748e+00, -1.319449057437554e+00, -9.147271426076817e-01, -9.152973322571392e-01, -2.297473004402688e-01, -2.296586859954191e-01, -1.167558105317267e-01, -3.930288248512680e-03, -4.290208970534545e-02, -1.734573695149444e-06, -3.727532785032675e-03, -3.453394724839816e-03, -7.789290808523586e-06, -7.204381327880205e-08]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_mpwb1k_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_mpwb1k", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.132698136106101e-04, 0.000000000000000e+00, -1.129483109645191e-04, -4.192202398223087e-04, 0.000000000000000e+00, -4.182247350310022e-04, -4.738031460854114e-02, 0.000000000000000e+00, -4.740894868448742e-02, 7.291247483411922e-03, 0.000000000000000e+00, 2.480687275996193e+01, -3.442848548250971e+01, 0.000000000000000e+00, 9.214031160352361e+02, 2.017237816205122e+01, 0.000000000000000e+00, 1.812842856131215e+01, 1.800385419802368e+04, 0.000000000000000e+00, 3.307209980201832e+02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_mpwb1k_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_mpwb1k", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_mpwb1k_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_mpwb1k", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-1.535420278481735e-03, -1.534572015769308e-03, -1.754667312298453e-03, -1.753913243294452e-03, -4.414736987621208e-04, -4.386980204772437e-04, -9.437213115668527e-02, -1.176431803685671e-07, -1.559459506086730e-02, -3.855053562272995e-11, -1.373116706943418e-07, -1.256638198788993e-07, -8.551510937470655e-12, -4.468127138692301e-12]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
