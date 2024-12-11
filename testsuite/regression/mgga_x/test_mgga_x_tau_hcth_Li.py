
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_tau_hcth_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_tau_hcth", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.922936431340191e+00, -1.335076396902001e+00, -3.204013965722211e-01, -1.737816519075953e-01, -6.887902948581857e-02, -4.383678922650817e-02, -7.004674153159432e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_tau_hcth_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_tau_hcth", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.562504405879161e+00, -2.564660311501459e+00, -1.779842301534704e+00, -1.781354322037648e+00, -4.272054319097305e-01, -4.270205460822773e-01, -2.317143056327197e-01, -5.558462358343627e-02, -9.182374355142364e-02, -1.770540632894647e-03, -5.869959109556963e-02, -5.801032899807369e-02, -1.182632464352400e-03, -2.418191073310126e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_tau_hcth_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_tau_hcth", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([3.477498872881161e-04, 0.000000000000000e+00, 3.467771375809786e-04, 1.495243320453563e-03, 0.000000000000000e+00, 1.490388775320483e-03, 4.028634720578149e-01, 0.000000000000000e+00, 4.030558040579517e-01, 5.077869697994476e+00, 0.000000000000000e+00, -1.547195723462817e+00, 1.811592604548151e+02, 0.000000000000000e+00, -9.890885621878859e+00, -6.588500757320979e-04, 0.000000000000000e+00, -1.468286722710493e+00, -4.514310421269626e-10, 0.000000000000000e+00, 4.085184691569540e+12])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_tau_hcth_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_tau_hcth", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-6.675520359922211e-05, -6.813851837810585e-05, 7.296983352085258e-05, 7.252330707972409e-05, -1.600383737389809e-05, -1.668056575659528e-05, -1.510882845594124e-03, -2.786127548641670e-08, -1.254084397991792e-04, -1.447982697407827e-14, -3.153359629317155e-13, -3.192933108956261e-08, 9.770874476322328e-24, -6.889780352066290e-20])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
