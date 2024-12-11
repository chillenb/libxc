
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_pbe_vwn_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_pbe_vwn", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-6.050876266203009e-02, -4.393151872810517e-02, -3.098911866846632e-03, -1.513599949677204e-02, -1.352793796091493e-03, -7.667159085573971e-09, -1.672923952145133e-16])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_pbe_vwn_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_pbe_vwn", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.165825301881833e-01, -1.164563192456599e-01, -1.020851207682327e-01, -1.019906594028526e-01, -1.604175298558849e-02, -1.604716874702639e-02, -2.417415141413246e-02, -1.022567624944596e-01, -6.597751456813768e-03, 3.430535763547381e-01, -4.963545409036289e-08, -4.988002792203598e-08, -1.035650243948888e-15, -1.234709762817227e-15])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_pbe_vwn_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_pbe_vwn", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([4.598657831051778e-05, 9.197315662103555e-05, 4.598657831051778e-05, 1.493543173532866e-04, 2.987086347065732e-04, 1.493543173532866e-04, 3.142805199055021e-03, 6.285610398110039e-03, 3.142805199055021e-03, 3.380775069368135e+00, 6.761550138736269e+00, 3.380775069368135e+00, 1.132598334653427e+01, 2.265196669306854e+01, 1.132598334653427e+01, 1.691259413572495e-04, 3.382518827108431e-04, 1.691259413572495e-04, 1.460311582676788e-06, 2.920812142470481e-06, 1.460311582676788e-06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
