
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_ktbm_2_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.999912578003352e+00, -1.410550206908387e+00, -3.376853238473614e-01, -1.791444872638165e-01, -7.426282437400758e-02, -1.248599922010074e-02, -2.286944139264936e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_ktbm_2_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.489851848047687e+00, -2.492276517431962e+00, -1.664931175051478e+00, -1.666381955538481e+00, -4.176968209500358e-01, -4.192271588741870e-01, -2.296791341644741e-01, -1.579478958074326e-02, -8.800996283743355e-02, -5.009326617805149e-04, -1.618027007873182e-02, -1.648850502506565e-02, -3.258236569853912e-04, -2.378659298561406e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_2_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-5.345836370376396e-04, 0.000000000000000e+00, -5.327462496787759e-04, -2.126016755828308e-03, 0.000000000000000e+00, -2.119856997862728e-03, -5.887260418466368e-02, 0.000000000000000e+00, -6.081050857252299e-02, -8.163393186768955e+00, 0.000000000000000e+00, -1.518078208158708e+01, -8.182262409989278e+01, 0.000000000000000e+00, -3.798990280568849e+04, -4.349406975428629e-01, 0.000000000000000e+00, -1.357412729964539e+01, -8.935358603191639e-01, 0.000000000000000e+00, -1.719859807085329e+05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_2_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([1.940311785941332e-02, 1.938708925562076e-02, 2.757394004367337e-02, 2.756415743499493e-02, 1.630195652972920e-02, 1.744615753125071e-02, 2.140198398062934e-01, 1.941292742206314e-04, 2.209203520116539e-01, 1.547848124495460e-05, 1.318542005998650e-07, 1.974894614315605e-04, 8.590581410386565e-16, 7.502582111197242e-06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
