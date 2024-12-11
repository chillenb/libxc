
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_xc_tpsslyp1w_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_tpsslyp1w", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.817772691168516e+00, -1.271148996349392e+00, -3.337358466409343e-01, 1.378812478611135e+164, 1.493259757059933e+166, 1.941281829501726e+218, -4.151527992541478e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_xc_tpsslyp1w_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_tpsslyp1w", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.421114134453294e+00, -2.423094496327556e+00, -1.694988338355532e+00, -1.696280233574959e+00, -4.358926164735905e-01, -4.355908996999010e-01, -1.297486557882976e+164, -3.209753412424897e+164, -1.854099720485020e+165, -1.419683830465333e+166, 7.389189376096441e+218, 7.450044050842161e+218, -6.193156747219857e-04, -3.689214931255680e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_xc_tpsslyp1w_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_tpsslyp1w", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-3.385912964314803e-04, 3.864883412170266e-06, -3.360184450499240e-04, -9.317694317665013e-04, 2.698736924043955e-05, -9.302460976158754e-04, -1.417096355447712e-02, 3.532584519053778e-02, -1.476480934803540e-02, -7.459481074705663e+00, 3.401139729395250e+00, 2.273874498182499e+00, -3.256405952267047e+01, 1.744135403403949e+01, 1.130451879999684e+01, 3.012497094755041e-02, 5.872712018115467e-02, -2.334805152723543e-01, -8.108568371384401e-11, 0.000000000000000e+00, -1.557397863841548e+11]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_xc_tpsslyp1w_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_tpsslyp1w", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [2.323678073304851e-03, 2.304817011437339e-03, 2.364145176797173e-03, 2.368349765299592e-03, -6.355538362972272e-04, -6.726087519172267e-04, 2.683296950295916e-02, 6.714990221518002e-11, -1.556610338874278e-02, 3.477076270561184e-17, 7.575171418605716e-16, 7.697046751037394e-11, 1.425570721901136e-33, -5.495081988232815e-13]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
