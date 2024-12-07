
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_xc_hcth_120_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_hcth_120", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-6.938521010680987e-01, -6.090774869454966e-01, -3.652840831500948e-01, -1.742805572229907e-01, -1.200649332090432e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_xc_hcth_120_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_hcth_120", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-9.234463391611584e-01, -1.275103284839625e-01, -8.178001435371118e-01, -1.700908037093183e-01, -4.409327837398530e-01, -1.832201339774105e-01, -1.263442943244669e-01, 5.070077759452025e-02, -1.588564520746891e-02, 3.849699496519807e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_xc_hcth_120_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_hcth_120", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [2.457160131294864e-02, 0.000000000000000e+00, -1.137257346767783e+21, 3.896842105819857e-03, 0.000000000000000e+00, -8.103253130933439e+20, -9.334248724504372e-02, 0.000000000000000e+00, -3.079542752912958e+20, -1.168800106510957e+01, 0.000000000000000e+00, 7.419589201645958e+19, -1.399175632635846e+01, 0.000000000000000e+00, 1.170137312013740e+14]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
