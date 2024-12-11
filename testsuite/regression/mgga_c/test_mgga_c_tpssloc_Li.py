
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_tpssloc_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_tpssloc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-9.347144678185146e-02, -8.371482262002407e-02, -4.959806172627841e-02, -1.808614940689015e-02, -1.095911360424974e-02, -8.827962574675379e-35, -8.300292557722951e-19]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_tpssloc_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_tpssloc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.026587664126353e-01, -1.024968785276771e-01, -9.254539378426754e-02, -9.240668603323736e-02, -5.664537600732751e-02, -5.668890672673786e-02, -2.101628026242187e-02, -1.243110661274556e-01, -1.310473963821066e-02, -7.152742107349912e-02, -6.434623397340148e-20, 3.469141197333486e-18, -5.643972820976097e-19, -5.039187317793476e-19]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_tpssloc_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_tpssloc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [7.796766773780362e-05, 1.559353753326970e-04, 7.796766776445851e-05, 3.354732300433228e-04, 6.709464600866456e-04, 3.354732300433228e-04, 1.009817185227372e-01, 2.019634370454744e-01, 1.009817185227372e-01, 2.344381142227896e+00, 4.688762288220602e+00, 2.344381824530762e+00, 9.451250511536134e+01, 1.890250102307227e+02, 9.451250511536134e+01, 0.000000000000000e+00, -2.736165374515218e-15, 0.000000000000000e+00, -9.838156340814792e-15, -6.053034208689736e-15, -7.084994418514591e-15]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_tpssloc_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_tpssloc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-1.686333919746283e-09, -1.686333919746282e-09, -2.808125991128571e-82, -2.808125991128570e-82, -8.864821255179366e-75, -8.864821255179364e-75, -3.798314469757896e-10, -3.798314469757056e-10, -2.940009702016773e-25, -2.940009699641517e-25, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
