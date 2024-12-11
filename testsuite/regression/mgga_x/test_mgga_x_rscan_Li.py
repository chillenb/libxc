
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_rscan_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_rscan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.975969814857509e+00, -1.311711341635397e+00, -2.343134284415891e-01, -1.813631382317139e-01, -6.955265857253906e-02, -4.816589062986553e-03, -1.021140385176997e-06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_rscan_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_rscan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.759869258033552e+00, -2.762442186600050e+00, -1.961916841969628e+00, -1.963401532757822e+00, -3.254542802364346e-01, -3.259536528436600e-01, -2.471670597958007e-01, -1.124692927991319e-02, -9.352479978643273e-02, -1.105765744346232e-04, -5.911150703878192e-03, -1.184713627483785e-02, -3.384378308926088e-06, -2.659634374410066e-22])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_rscan_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_rscan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-3.812009045244698e-04, 0.000000000000000e+00, -3.797763791130670e-04, -2.152687296114754e-03, 0.000000000000000e+00, -2.143624119018384e-03, -4.330921597419159e-01, 0.000000000000000e+00, -4.335858111173673e-01, -4.474322187319567e+00, 0.000000000000000e+00, 2.805042356183251e+01, -5.754862513862457e+01, 0.000000000000000e+00, 2.662357818440063e+04, 3.064038850387319e-01, 0.000000000000000e+00, 2.519790219596326e+01, 1.940994653209789e-02, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_rscan_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_rscan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([1.802808897161596e-02, 1.801020311666251e-02, 3.127429511578545e-02, 3.123622249454146e-02, 2.388765828176336e-03, 2.519391006062151e-03, 1.620341584960541e-01, 4.526913255355321e-36, 6.074836054441127e-02, 1.818089887113042e-54, 8.302996330573498e-33, 2.626418651116829e-36, 1.274175366634189e-44, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
