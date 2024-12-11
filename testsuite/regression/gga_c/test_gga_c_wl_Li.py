
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_wl_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_wl", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-6.340210328627729e-02, -4.506375346503726e-02, 8.172097280961109e-04, -5.658422389623594e-05, -7.523513271664298e-09, 2.399118889924541e-02, 2.097474678179833e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_wl_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_wl", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.350802279093475e-01, -1.347281355837670e-01, -1.092308486808315e-01, -1.089901226213149e-01, -2.823258912262019e-02, -2.822859015228113e-02, -4.672458656484329e-05, -3.064471416122383e-01, -1.024794488594066e-07, -8.078065871397574e-02, 2.035522850699090e-02, 2.227945841594099e-02, 2.166566682665424e-03, 6.726881197878459e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_wl_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_wl", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([1.037565953417690e-04, 3.573366904707737e-05, 1.032177561214140e-04, 2.731856972084236e-04, 1.237069855137871e-04, 2.718753147986158e-04, 6.817957471275482e-03, 1.453290031690368e-02, 6.817290113196700e-03, 8.809893924536879e-03, 1.721731624415613e-02, 2.126120838169432e+03, 2.160103085670883e-04, 4.319615754402981e-04, 4.464246219814317e+07, -1.078362050044090e+02, 2.754249719636056e+02, -1.163151133753399e+02, 1.345239898784630e+07, 9.389961594565177e+07, -3.523091008933492e+08])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
