
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_xc_b88b95_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_b88b95", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.342938150842850e+00, -9.677256650479334e-01, -3.208931345694241e-01, -1.157203421554524e-01, -5.850896118057283e-02, -9.592136249475075e-02, -3.860320143300076e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_xc_b88b95_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_b88b95", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.677399448729182e+00, -1.678723779929563e+00, -1.158480208299401e+00, -1.159268948311831e+00, -2.732883235096047e-01, -2.731181051135281e-01, -1.495875566104452e-01, -2.659431448138183e-02, -5.392713528356109e-02, -5.562661315718601e-03, -2.650975645897359e-02, -2.653919697791519e-02, -5.379724232411123e-03, -4.646231027185901e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_b88b95_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_b88b95", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.559315855131522e-04, 0.000000000000000e+00, -1.554675219020174e-04, -5.825139728150069e-04, 0.000000000000000e+00, -5.810009703687249e-04, -7.286257622794141e-02, 0.000000000000000e+00, -7.291632201765413e-02, -6.725000791491875e-01, 0.000000000000000e+00, -9.588559462871181e+02, -4.862761995333354e+01, 0.000000000000000e+00, -3.492152052788053e+07, -8.377408208466244e+02, 0.000000000000000e+00, -8.399613503546587e+02, -1.036623715792920e+08, 0.000000000000000e+00, -3.088520062583042e+08])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_b88b95_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_b88b95", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-1.535420278481735e-03, -1.534572015769308e-03, -1.754667312298453e-03, -1.753913243294452e-03, -4.414736987621208e-04, -4.386980204772437e-04, -9.437213115668527e-02, -1.176431803685671e-07, -1.559459506086730e-02, -3.855053562272995e-11, -1.373116706943418e-07, -1.256638198788993e-07, -8.551510937470655e-12, -4.468127138692301e-12])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
