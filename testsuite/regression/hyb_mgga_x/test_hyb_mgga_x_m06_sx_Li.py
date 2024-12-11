
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_x_m06_sx_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m06_sx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.067919224818213e+00, -7.868877927246539e-01, -1.065637375655665e-01, -9.407627268302104e-02, -3.423732842242482e-02, -2.167970675152681e-02, -3.688840570173809e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_x_m06_sx_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m06_sx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.243878024224076e+00, -1.245225619842230e+00, -9.990551864569429e-01, -9.992555836567567e-01, -2.680735679913861e-01, -2.717739103571096e-01, -1.280773990893943e-01, -2.737349701640090e-02, -6.366188376572024e-02, -8.767626108867393e-04, -2.906446680504325e-02, -2.855832944228061e-02, -5.856417718957299e-04, -2.307789698601301e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_m06_sx_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m06_sx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-3.094495521274814e-04, 0.000000000000000e+00, -3.083243208395038e-04, -1.390063552148417e-03, 0.000000000000000e+00, -1.385203441994246e-03, -3.857916436404667e-01, 0.000000000000000e+00, -3.874753517195605e-01, -4.543001407209698e+00, 0.000000000000000e+00, -7.839351174135163e-01, -1.955191925575787e+02, 0.000000000000000e+00, -5.031260735701854e+00, -3.351146932226172e-04, 0.000000000000000e+00, -7.437732719289286e-01, -2.296472429353689e-10, 0.000000000000000e+00, -3.701072400948956e+12]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_m06_sx_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m06_sx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-2.550253115457866e-02, -2.547729756821583e-02, -7.270323007301334e-03, -7.362760648229427e-03, 2.310852638524539e-02, 2.393490913917822e-02, 7.424722764958036e-02, -1.625832863081910e-05, 1.508147932265174e-01, -3.334000908474619e-09, -8.094137563505429e-09, -1.754669136917268e-05, -4.535039279001429e-20, -1.953087966582152e-09]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
