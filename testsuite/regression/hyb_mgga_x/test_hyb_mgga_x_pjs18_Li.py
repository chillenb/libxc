
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_x_pjs18_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_pjs18", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.754350995222188e+00, -1.172132627586892e+00, -1.981729908594685e-01, -4.559395853484988e-02, -4.125744430547497e-03, 1.011582958234584e-03, 1.925950250226003e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_x_pjs18_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_pjs18", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.322683414763438e+00, -2.325073564858407e+00, -1.510995019412049e+00, -1.512552430852698e+00, -1.929538850950943e-01, -1.940453464333592e-01, -8.071507865135179e-02, -6.291471206138412e-06, -7.420678877696095e-03, -8.757029976648602e-10, 2.380566883282505e-03, -6.509191303997562e-06, 2.618966604915218e-05, -9.501734640633163e-11]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_pjs18_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_pjs18", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.520027868601776e-04, 0.000000000000000e+00, -2.510896469297390e-04, -1.029785668990431e-03, 0.000000000000000e+00, -1.026259655777535e-03, -8.661532563935292e-02, 0.000000000000000e+00, -8.591059081013727e-02, -4.033825119366526e-01, 0.000000000000000e+00, -3.390147726054458e+00, -2.105589931617702e+00, 0.000000000000000e+00, -3.947014622339227e+00, -6.034546072096808e+00, 0.000000000000000e+00, -3.358158242075673e+00, -6.164290900443538e+01, 0.000000000000000e+00, -3.951245111238042e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_pjs18_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_pjs18", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_pjs18_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_pjs18", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [9.925348589918953e-03, 9.919811691779457e-03, 1.205107996812094e-02, 1.204540740326336e-02, 1.052536483281353e-02, 1.050523671659572e-02, 1.856445042087568e-02, 1.352961758117568e-05, 2.207131890582382e-03, 4.901867144443737e-10, 1.561685588393940e-05, 1.526851349137438e-05, 1.461485197450178e-10, 5.252828130920616e-11]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
