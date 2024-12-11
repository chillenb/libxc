
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_rmggac_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_rmggac", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-7.381800500678835e-16, -3.051981233820194e-02, -2.514861924389029e-02, -1.306221909917834e-02, -1.542268767770037e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_rmggac_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_rmggac", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-7.403014102181902e-17, -2.198763077908849e-01, -3.052408271797261e-02, -2.460633966535984e-01, -2.897617081963829e-02, -1.954171001819024e-01, -1.550489257307257e-02, -9.115272106719999e-02, -1.966577640593631e-03, -7.182517282212357e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_rmggac_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_rmggac", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [2.885778129383089e-17, 5.771556258766178e-17, 2.885778129383089e-17, 3.130804893794320e-02, 6.261609787588640e-02, 3.130804893794320e-02, 2.184515207896490e-01, 4.369030415792980e-01, 2.184515207896490e-01, 4.285652118567145e+01, 8.571304237134291e+01, 4.285652118567145e+01, 8.748331548386607e+06, 1.749666309677321e+07, 8.748331548386607e+06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_rmggac_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_rmggac", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-5.184761820970873e-31, -5.123978193499695e-31, -4.881268052723167e-03, -4.801697884462101e-03, 3.555355129317109e-04, 3.542436959496998e-04, 4.398224395666031e-05, 4.398110458026660e-05, 6.713250421384381e-09, 6.713250471617984e-09]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
