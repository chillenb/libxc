
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_hyb_tau_hcth_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_hyb_tau_hcth", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.033524715507457e-02, -2.237065362963215e-02, -1.475138760411164e-02, -7.526127618752529e-03, -8.939841649956265e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_hyb_tau_hcth_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_hyb_tau_hcth", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.267808549632830e-03, -2.071815900032083e-01, -4.001707929304159e-02, -2.244708155924317e-01, -2.027668911152228e-02, -2.021285439846399e-01, -8.840016460885145e-03, 7.000778424721528e-02, -1.139917126525879e-03, 3.512203291414927e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_hyb_tau_hcth_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_hyb_tau_hcth", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.584951091145935e-01, 0.000000000000000e+00, -6.300984258355794e+20, 6.894493166383046e-03, 0.000000000000000e+00, -4.459670302342136e+20, 7.542605357330780e-03, 0.000000000000000e+00, -1.520307628417418e+20, -8.108738538233733e-03, 0.000000000000000e+00, 8.109162927051750e+19, -1.559326176529540e-02, 0.000000000000000e+00, 7.322743318743720e+13]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
