
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_r2scan01_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_r2scan01", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.973315652811421e+00, -1.309794209194168e+00, -2.342832281820834e-01, -1.808641809759491e-01, -5.244988779359668e-02, -4.816589062986550e-03, -1.021140385176995e-06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_r2scan01_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_r2scan01", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.757525522850051e+00, -2.760062739973662e+00, -1.957398049816152e+00, -1.958904715081391e+00, -3.254056935755857e-01, -3.259033195562960e-01, -2.474647323528177e-01, 2.665779724507994e-02, -7.716512692898699e-02, 2.403912683000811e-04, 1.317774877734327e-02, 2.808980546488996e-02, 7.105915596221665e-06, -2.491504539376022e-22])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_r2scan01_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_r2scan01", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-3.516451606564670e-04, 0.000000000000000e+00, -3.502902721196391e-04, -1.794445173811324e-03, 0.000000000000000e+00, -1.787870423224121e-03, -1.128463384183379e-02, 0.000000000000000e+00, -1.186032304880274e-02, -4.628848635402363e+00, 0.000000000000000e+00, -9.427722630567966e+02, -2.649430351565686e+01, 0.000000000000000e+00, -6.851494983035305e+05, -8.361959108812568e+00, 0.000000000000000e+00, -8.507493305060365e+02, -4.629081269723348e-01, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_r2scan01_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_r2scan01", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([1.819536247936829e-02, 1.817435574935419e-02, 3.098683975490197e-02, 3.095264261266209e-02, 2.378177799997483e-03, 2.508153711235639e-03, 1.773829230416201e-01, 1.239905555477703e-02, 6.055273548559918e-02, 2.900015584533793e-04, 1.287353603991371e-04, 1.272706039871861e-02, 5.856115742162687e-11, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
