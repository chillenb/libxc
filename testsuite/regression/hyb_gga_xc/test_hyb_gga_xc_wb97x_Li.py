
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_wb97x_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_wb97x", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.432422229186551e+00, -9.885129016637164e-01, -2.416397390544978e-01, -4.974022326724786e-02, -8.157534713564008e-03, 1.515523898071225e-02, 2.583037171310346e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_wb97x_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_wb97x", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.864989926807422e+00, -1.866725087713755e+00, -1.211748806562764e+00, -1.212804224350054e+00, -3.225620393588756e-02, -3.140654804388085e-02, -9.684430970792780e-02, 6.120556059499347e-01, -5.017726686926794e-03, 3.725293492417506e-01, 1.876930904477451e-02, 1.940237340150076e-02, 1.521188145278077e-04, 8.790963561858691e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_wb97x_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_wb97x", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.282353459310364e-04, 0.000000000000000e+00, -1.275784171169587e-04, -7.372217088186609e-04, 0.000000000000000e+00, -7.345024364923892e-04, -1.688074059281209e-01, 0.000000000000000e+00, -1.692966176725539e-01, 7.532923604632618e+00, 0.000000000000000e+00, 9.171183137753702e+01, -1.913225748365556e+01, 0.000000000000000e+00, 1.082813567445394e+04, 1.057892057356316e+00, 0.000000000000000e+00, 1.129342791776935e+00, 1.757087434860222e+00, 0.000000000000000e+00, 2.849149001350657e+01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
