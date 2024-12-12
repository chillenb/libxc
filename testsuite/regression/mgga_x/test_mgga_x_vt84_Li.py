
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_vt84_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_vt84", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.998999629457416e+00, -1.385715869566933e+00, -3.784564473938850e-01, -1.791942364555329e-01, -7.399995373851881e-02, -1.067975072634151e-02, -6.423418311397198e-11])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_vt84_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_vt84", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.355105303075060e+00, -2.357261990912713e+00, -1.670536136809925e+00, -1.672293214186750e+00, -3.248760328149582e-01, -3.262080594618864e-01, -2.098866136279890e-01, -1.528597631044639e-02, -7.762078621129156e-02, -1.189923482350709e-09, -1.597465079041436e-02, -1.590218259885505e-02, -3.025281169602656e-10, -1.300429553593972e-10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_vt84_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_vt84", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.418101765251546e-03, 0.000000000000000e+00, -1.416436836556099e-03, -2.600865446789518e-03, 0.000000000000000e+00, -2.596398214438739e-03, -9.414525913325482e-02, 0.000000000000000e+00, -9.323769350791652e-02, -3.718641591473332e+01, 0.000000000000000e+00, 1.697306963912622e+01, -6.061298058283544e+01, 0.000000000000000e+00, 6.033159137332542e-01, 1.384910096869943e+01, 0.000000000000000e+00, 1.427925674679184e+01, 4.391537261942146e-01, 0.000000000000000e+00, 6.286025293795701e-01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_vt84_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_vt84", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([7.374772048203662e-02, 7.389912572326827e-02, 4.642363974218308e-02, 4.653559094356929e-02, 2.647362694853223e-03, 2.594838165098463e-03, 1.345934458226161e+00, -5.692649747610254e-09, 5.571102251935036e-02, -1.694857481296172e-17, -1.353303417086486e-16, -5.787831298205824e-09, -3.817879022830392e-42, -2.821485406670394e-19])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
