
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_ityh_pbe_Li_restr_1_zk():
    # Prepare the input
    inp = test_data["Li_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_ityh_pbe", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.606391390534643e+00, -1.090659265128276e+00, -1.806078718312920e-01, -2.532543796337502e-02, -2.067729581340455e-03, -2.644101939118950e-05, -1.487849555803382e-10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_ityh_pbe_Li_restr_1_vrho():
    # Prepare the input
    inp = test_data["Li_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_ityh_pbe", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.076215351367027e+00, -1.365234758908073e+00, -2.549759428828629e-01, -4.557777565018507e-02, -4.046197913172705e-03, -5.285133226701500e-05, -2.975698566383762e-10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_ityh_pbe_Li_restr_1_vsigma():
    # Prepare the input
    inp = test_data["Li_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_ityh_pbe", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.076114204297804e-04, -3.929787893721138e-04, -9.186424135560058e-03, -2.561571035829283e-01, -7.168148926641173e-02, -3.050579569571063e-07, -1.586143934829315e-13]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
