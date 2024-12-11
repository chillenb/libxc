
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_mvsbs_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mvsbs", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.206952016306527e+00, -1.430802047308850e+00, -2.854570290532622e-01, -1.909791046253957e-01, -6.048165911402356e-02, -2.025550296220105e-03, -4.369556749392754e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_mvsbs_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mvsbs", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.464929526219913e+00, -2.467094298626298e+00, -2.390143888414805e+00, -2.395348651501566e+00, -3.744226548944648e-01, -3.739891767679742e-01, -2.377987292567359e-01, -5.134101828955676e-03, -7.785881904916415e-02, -3.655839044042649e-05, -2.097895765112826e-03, -5.440379054354206e-03, -1.017461541350278e-06, -2.183783667929595e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mvsbs_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mvsbs", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [9.603430357855581e-04, 0.000000000000000e+00, 9.571013129012537e-04, -1.884832928852488e-03, 0.000000000000000e+00, -1.897154281674792e-03, -3.099469614235131e-02, 0.000000000000000e+00, -3.166060499687030e-02, 1.050230639561630e+01, 0.000000000000000e+00, 7.225919899845833e+00, -2.452548454522213e+01, 0.000000000000000e+00, 4.075391665872448e+03, 5.236555288304663e-02, 0.000000000000000e+00, 6.557017331256243e+00, 2.571415445732135e-03, 0.000000000000000e+00, -3.248074647439588e+05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mvsbs_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mvsbs", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-6.788161649643526e-02, -6.780391288559828e-02, 7.071336278578104e-02, 7.115186837053356e-02, -1.149939251269486e-03, -1.206644761908495e-03, -4.814656259887743e-01, 1.344453334218985e-04, -2.330537873240474e-02, 2.417742381014042e-06, 1.132381472700379e-06, 1.387952066021707e-04, 4.546002882282868e-13, -1.687526639960317e-12]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
