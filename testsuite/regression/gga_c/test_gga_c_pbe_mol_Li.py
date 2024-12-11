
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_pbe_mol_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_pbe_mol", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-5.524044031711734e-02, -3.853925451254270e-02, -2.130294048116127e-03, -1.450679294046878e-02, -9.594100864888429e-04, -4.820982309541866e-09, -1.156843718042033e-16])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_pbe_mol_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_pbe_mol", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.141551872006747e-01, -1.140363359582243e-01, -9.754966264277033e-02, -9.746285328748348e-02, -1.147255410766256e-02, -1.147657654446541e-02, -2.471963421827451e-02, -9.072842799536607e-02, -4.946621536726696e-03, 2.663216290566559e-01, -3.120389535552954e-08, -3.136064079160468e-08, -7.214077967811205e-16, -8.546562437795890e-16])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_pbe_mol_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_pbe_mol", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([4.885432917280109e-05, 9.770865834560218e-05, 4.885432917280109e-05, 1.527626377131914e-04, 3.055252754263828e-04, 1.527626377131914e-04, 2.274464955049411e-03, 4.548929910098821e-03, 2.274464955049411e-03, 4.023063979992194e+00, 8.046127959984386e+00, 4.023063979992194e+00, 8.628641582823962e+00, 1.725728316564792e+01, 8.628641582823962e+00, 1.063559099079353e-04, 2.127118198217388e-04, 1.063559099079353e-04, 1.017855851921395e-06, 2.035444796919866e-06, 1.017855851921395e-06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
