
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_lspbe_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_lspbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.794330407429061e+00, -1.283441598996950e+00, -4.134299395134037e-01, -1.600162652415940e-01, -8.029779133214571e-02, -3.930914587860388e-04, 1.869231343926682e-10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_lspbe_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_lspbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.242839774576198e+00, -2.244972931169177e+00, -1.519572500224318e+00, -1.520940820744590e+00, -4.062167886318486e-01, -4.064109210602051e-01, -2.053175508922509e-01, -3.427556249986370e-03, -7.694321435421426e-02, 3.462497607774407e-09, -5.097037701891521e-03, -4.355573446683101e-03, 8.803635661218664e-10, 3.784283237686813e-10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_lspbe_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_lspbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-2.546235985993394e-04, 0.000000000000000e+00, -2.537475246789779e-04, -1.006303014529053e-03, 0.000000000000000e+00, -1.003078970277782e-03, -7.033495594171192e-02, 0.000000000000000e+00, -7.014203749381352e-02, -3.945369940097592e+00, 0.000000000000000e+00, 2.956867180542450e+01, -6.591072288003100e+01, 0.000000000000000e+00, -1.755509876018619e+00, 3.764303295090064e+01, 0.000000000000000e+00, 3.195616170518999e+01, -1.277946409421466e+00, 0.000000000000000e+00, -1.829248784135140e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
