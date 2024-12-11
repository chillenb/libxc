
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_m05_2x_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m05_2x", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-9.339682481736633e-02, -8.371482262002408e-02, -4.959806172627838e-02, -1.766128863017501e-02, -1.095898628587277e-02, 3.089528642344976e-02, 5.320218249785507e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_m05_2x_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m05_2x", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.028042696298762e-01, -1.024833015888350e-01, -9.254539378426756e-02, -9.240668603323737e-02, -5.664537600732748e-02, -5.668890672673783e-02, -2.092425139254580e-02, 1.120109720539339e+00, -1.310470937105105e-02, 6.752767207717643e-01, 5.615859021885360e-02, 5.720530075587522e-02, 9.763237873998013e-04, 1.568035563080206e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_m05_2x_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m05_2x", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [5.610429611283492e-04, 0.000000000000000e+00, 5.623413065489033e-04, 2.543225714720553e-03, 0.000000000000000e+00, 2.534765984555303e-03, 1.476990104695446e+00, 0.000000000000000e+00, 1.480518752334079e+00, 4.652733598414218e+01, 0.000000000000000e+00, -1.974122829347190e+02, 1.566668784257689e+03, 0.000000000000000e+00, -1.687297131348152e+06, -7.803610204480992e+00, 0.000000000000000e+00, -3.748501782636307e+02, -2.654150330461250e+01, 0.000000000000000e+00, 9.263178300261492e+13]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_m05_2x_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m05_2x", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-1.517867435606015e-05, -7.577336033803976e-46, -6.416391505146912e-45, -6.395586446153691e-45, -3.028647150211534e-41, -3.192749052890911e-41, -2.203149546907371e-35, 5.413404553452415e-03, -2.985309460798631e-34, 6.983701417806610e-04, 1.159620014785305e-04, 5.446451088939682e-03, 3.222564607737553e-09, -2.553864405425215e-23]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
