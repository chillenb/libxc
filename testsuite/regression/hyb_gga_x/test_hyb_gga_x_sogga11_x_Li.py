
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_x_sogga11_x_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_x_sogga11_x", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.055733710231325e+00, -7.410591105187676e-01, -1.142516360087545e-01, -9.485394229435561e-02, -3.804756999115957e-02, 4.919664398672417e-06, 1.070759781293106e-08])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_x_sogga11_x_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_x_sogga11_x", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.370904324274664e+00, -1.372148156743534e+00, -9.512295725769295e-01, -9.520367085762336e-01, -5.449506167300675e-01, -5.452849283408515e-01, -1.242774001413050e-01, 1.572006707716471e-05, -6.783793233669344e-02, 2.426327688334881e-08, 1.824247951891783e-05, 1.732530599446454e-05, 1.568168399967498e-08, 1.105758420860958e-08])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_x_sogga11_x_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_x_sogga11_x", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-6.189934320443279e-05, 0.000000000000000e+00, -6.171829359342050e-05, -1.919728359501374e-04, 0.000000000000000e+00, -1.914459612801431e-04, 1.902451426219319e-01, 0.000000000000000e+00, 1.904787027780659e-01, -1.088290347408390e+00, 0.000000000000000e+00, -9.505819638219594e-02, 3.743591593928175e+01, 0.000000000000000e+00, -6.431894737024638e-01, -9.596360114350677e-02, 0.000000000000000e+00, -8.988786947725809e-02, -4.682589817773152e-01, 0.000000000000000e+00, -6.702787320269961e-01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
