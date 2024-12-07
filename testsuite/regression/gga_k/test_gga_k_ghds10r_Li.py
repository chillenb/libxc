
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_ghds10r_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_ghds10r", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [2.016063220859260e+01, 1.181807803827240e+01, 3.700765148920173e+00, 1.836356203314823e-01, 1.079599571025859e-01, 3.069854398322075e+00, 1.268174499514641e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_ghds10r_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_ghds10r", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [2.228551760382988e+01, 2.233325781979334e+01, 8.713165586996460e+00, 8.732701564457663e+00, -2.425664261771462e+00, -2.435884182849413e+00, 2.351061806678791e-01, -3.066056340824317e+00, -1.564219275635826e-02, -1.283735335848137e+00, -3.040104331769165e+00, -3.146061812893159e+00, -1.499378999281771e+00, -1.272795346374771e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_ghds10r_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_ghds10r", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [1.927193223082906e-02, 0.000000000000000e+00, 1.921993237398214e-02, 5.758341797473614e-02, 0.000000000000000e+00, 5.743663279313743e-02, 4.153406735139716e+00, 0.000000000000000e+00, 4.158710224823615e+00, 2.605032494947879e+01, 0.000000000000000e+00, 7.829811571772349e+04, 4.181517806762772e+02, 0.000000000000000e+00, 2.454376727176486e+09, 6.733474755479385e+04, 0.000000000000000e+00, 6.882557364072799e+04, 8.236143114997760e+09, 0.000000000000000e+00, 2.292358375079883e+10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
