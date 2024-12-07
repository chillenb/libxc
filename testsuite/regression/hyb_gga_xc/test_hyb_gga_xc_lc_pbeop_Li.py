
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_lc_pbeop_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_lc_pbeop", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.660344581531115e+00, -1.137163827862788e+00, -1.938453012476010e-01, -4.393574647470831e-02, -4.030966499293993e-03, -6.919959086978611e-04, -1.075262710257885e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_lc_pbeop_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_lc_pbeop", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.144680323992946e+00, -2.146611801449594e+00, -1.433724590629073e+00, -1.434890812693953e+00, -2.826910630220372e-01, -2.826774659525359e-01, -7.594725534263604e-02, -4.498133522009033e-01, -7.764810158368082e-03, -2.111742998891178e+01, -9.251416034378602e-04, -9.398593495686042e-04, -1.056591950575925e-05, -2.482284699737816e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_lc_pbeop_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_lc_pbeop", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.936799740992494e-04, 0.000000000000000e+00, -1.931430628911624e-04, -6.844708085199316e-04, 0.000000000000000e+00, -6.828761544635253e-04, -1.234903004933831e-02, 0.000000000000000e+00, -1.229175981791059e-02, -3.948306243146084e-01, 0.000000000000000e+00, 3.053588069336244e+01, -2.290836317811119e-01, 0.000000000000000e+00, 3.549267517993954e+05, 1.836202510395562e-02, 0.000000000000000e+00, 1.778327964382570e-02, 3.780651655453009e-02, 0.000000000000000e+00, 2.980284105430217e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
