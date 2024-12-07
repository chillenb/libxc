
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_tm_pbe_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_tm_pbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-6.242338989795505e-02, -4.204578890536159e-02, -1.682031816936040e-03, -1.997064484103751e-02, -1.766553936153176e-03, -8.968843149929906e-09, -2.851993814723119e-16]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_tm_pbe_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_tm_pbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.481659144896319e-01, -1.480204311779751e-01, -1.234006070051231e-01, -1.232971049496025e-01, -1.017357187055772e-02, -1.017702300937304e-02, -2.070101495927371e-02, -1.534833850815314e-01, -1.059427259419459e-02, 1.311593632448937e-01, -5.740509991134280e-08, -5.769815719204225e-08, -1.803062094161284e-15, -2.134069017421109e-15]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_tm_pbe_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_tm_pbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [7.499158501258619e-05, 1.499831700251725e-04, 7.499158501258619e-05, 2.194637851522122e-04, 4.389275703044241e-04, 2.194637851522122e-04, 2.125434055092603e-03, 4.250868110185183e-03, 2.125434055092603e-03, -1.225194168685310e+00, -2.450388337370620e+00, -1.225194168685310e+00, 1.987374283385260e+01, 3.974748566770521e+01, 1.987374283385260e+01, 1.980517291336416e-04, 3.961034582981927e-04, 1.980517291336416e-04, 2.550191210682959e-06, 5.100586828598160e-06, 2.550191210682959e-06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
