
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_revtca_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_revtca", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-6.124373773110634e-02, -4.871961162311641e-02, -4.762806912202816e-03, -9.644131965182677e-03, -5.514277022500154e-04, -2.500348716277619e-07, -9.853617544783627e-13]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_revtca_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_revtca", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-9.191742967999741e-02, -9.180711358688554e-02, -9.365254958982744e-02, -9.356964932961565e-02, -1.850337528246210e-02, -1.850742712864611e-02, -1.850695710525739e-02, -2.875432747655539e-01, -2.725600649200966e-03, -2.033886847930798e-01, -1.095170777661395e-06, -1.098823827963191e-06, -3.932721553904315e-12, -5.497348859843636e-12]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_revtca_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_revtca", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [2.372102317520150e-05, 4.744204635040299e-05, 2.372102317520150e-05, 1.098466498676712e-04, 2.196932997353424e-04, 1.098466498676712e-04, 3.137688002692894e-03, 6.275376005385788e-03, 3.137688002692894e-03, 3.658077986524286e+00, 7.316155973048573e+00, 3.658077986524286e+00, 4.834203050681829e+00, 9.668406101363658e+00, 4.834203050681829e+00, 3.172880419316142e-03, 6.345760838632284e-03, 3.172880419316142e-03, 5.092093770230665e-03, 1.018418754046133e-02, 5.092093770230665e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
