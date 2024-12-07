
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_2d_b86_mgc_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_2d_b86_mgc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-3.876787108619319e+00, -2.285610606970134e+00, -3.632200995658275e-01, -1.105811698355892e-01, -3.917159324816444e-02, -1.191474690884129e-02, -4.113751613994762e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_2d_b86_mgc_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_2d_b86_mgc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-5.687601885474147e+00, -5.695438321053176e+00, -3.240649469486107e+00, -3.244913713691188e+00, -4.398077820163170e-01, -4.397035404247804e-01, -1.498681420550162e-01, -9.987729680128508e-03, -4.714243098000405e-02, -4.415354613245427e-04, -1.047019074978691e-02, -1.045968353302859e-02, -3.358455522744965e-04, -2.473780599568692e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_2d_b86_mgc_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_2d_b86_mgc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.884654173032178e-04, 0.000000000000000e+00, -1.877347670991472e-04, -8.694841390744569e-04, 0.000000000000000e+00, -8.664506046881177e-04, -4.530894290602906e-02, 0.000000000000000e+00, -4.522326297331981e-02, -6.943242414015516e+00, 0.000000000000000e+00, -6.092817006530683e+01, -2.259222743108690e+01, 0.000000000000000e+00, -2.875949400730233e+05, -5.474584174769034e+01, 0.000000000000000e+00, -5.423896679902663e+01, -6.330071532016377e+05, 0.000000000000000e+00, -1.560405550657444e+06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
