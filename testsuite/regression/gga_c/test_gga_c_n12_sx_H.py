
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_n12_sx_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_n12_sx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-8.428221037190137e-02, -4.932509792478227e-02, -6.499044855646023e-03, 1.792686043349041e-02, 2.398138897573056e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_n12_sx_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_n12_sx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-9.782678970368601e-02, -2.153828801115608e-01, -1.342215939067512e-01, -2.281883406280839e-01, -8.687330287377371e-02, -1.998118832099368e-01, 1.513797649675520e-02, 1.053574411737858e-01, 3.056646502068034e-03, -1.730202700368332e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_n12_sx_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_n12_sx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [4.945121533697288e-02, 0.000000000000000e+00, -5.328505495818188e+20, 3.656257780551148e-02, 0.000000000000000e+00, -3.754209205870337e+20, 1.735387685359975e-01, 0.000000000000000e+00, -1.010431559055932e+20, 6.855008744756619e-01, 0.000000000000000e+00, 5.737375459778022e+19, 1.005736323035810e+00, 0.000000000000000e+00, -2.164242599354486e+14]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
