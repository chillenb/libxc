
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_m06_l_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m06_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.533382862314314e-02, -4.622040871474902e-02, -2.186102891279364e-01, 9.351737628926163e-03, -4.729206484086130e-02, 3.020445264932984e-02, 5.300910308403563e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_m06_l_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m06_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [1.672107347171720e-02, 1.667755296494183e-02, 2.462222710083022e-02, 2.468200674950948e-02, -8.133818181386121e-02, -7.807299983186014e-02, 2.000395325495654e-02, 1.113368291461979e+00, 1.004846898910654e-02, 6.607666333246419e-01, 6.710738506899179e-02, 6.801629219551623e-02, 1.364039553163481e-03, 1.582798583676456e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_m06_l_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m06_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-5.464766996703017e-04, 0.000000000000000e+00, -5.526174092770445e-04, -2.256528836577668e-03, 0.000000000000000e+00, -2.250522615531318e-03, -6.008096204913768e-01, 0.000000000000000e+00, -6.244369835366090e-01, -4.603691314167206e+01, 0.000000000000000e+00, -5.587614607053010e+02, -1.090326292579636e+02, 0.000000000000000e+00, -2.873454353543778e+06, -1.316989828431641e+01, 0.000000000000000e+00, -6.316428444049054e+02, -4.477853444440963e+01, 0.000000000000000e+00, -1.510121172778053e+13]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_m06_l_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m06_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-6.281605477065741e-03, -6.285977325580882e-03, -1.107331498650107e-02, -1.105581537321002e-02, -3.086151997191825e-02, -3.179033878824474e-02, -2.694618651610135e-01, 9.463721829536970e-03, -5.576376457520424e-01, 1.179505579487889e-03, 1.956440470619446e-04, 9.177501623174222e-03, 5.436832971552850e-09, 6.588483284176064e-09]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
