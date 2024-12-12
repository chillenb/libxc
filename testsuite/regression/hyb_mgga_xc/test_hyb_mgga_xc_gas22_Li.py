
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_xc_gas22_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_gas22", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.634982648674824e+00, -1.045719057410006e+00, -2.104956253307045e-01, 3.542338322564220e-02, -1.235937166981149e-02, -6.653851774344402e-02, -9.500613254689257e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_xc_gas22_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_gas22", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.339921582749688e+00, -2.341495617136224e+00, -1.449867300350700e+00, -1.451038456711246e+00, -2.183432923959838e-01, -2.181957817308378e-01, 9.335743897689493e-02, -1.450058156738497e-01, 1.250112629041875e-04, 1.339621303697835e+00, -4.725735717430857e-02, -4.502607125560982e-02, -3.258102562052846e-03, -9.091094982802091e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_gas22_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_gas22", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.207602129084790e-04, 0.000000000000000e+00, -1.210031465231726e-04, -5.484341548747322e-04, 0.000000000000000e+00, -5.470708123100113e-04, -3.539638306782936e-03, 0.000000000000000e+00, -3.169018503391733e-03, -7.360093797550565e+00, 0.000000000000000e+00, 1.974881650153797e+04, -7.821086800429430e-01, 0.000000000000000e+00, 1.003965642699503e+10, -3.044929953285796e+02, 0.000000000000000e+00, -3.227991684858498e+02, -8.733827252815237e+06, 0.000000000000000e+00, -1.338733907563332e+08])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_gas22_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_gas22", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([2.277406406948381e-02, 2.278338863998651e-02, 1.509821453158568e-02, 1.510022208146643e-02, -1.536148537017466e-02, -1.577093379468730e-02, -1.925526789142438e+00, -3.792391381616318e-05, -1.505235943251212e-01, -1.747324349741662e-06, -9.324319560744014e-08, -2.093904605986554e-04, -3.715438110662188e-18, -1.181364717057424e-07])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
