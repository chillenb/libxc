
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_xc_hcth_407_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_hcth_407", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-7.082789316525752e-01, -6.130740391221510e-01, -3.688815998627066e-01, -1.822782277402872e-01, -1.724662009599113e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_xc_hcth_407_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_hcth_407", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-9.434352282624223e-01, -1.498696236236283e-01, -8.220805983686933e-01, -1.734147767737678e-01, -4.490027425732727e-01, -1.658863940373028e-01, -7.337708233359035e-02, -2.086783603206976e-02, -2.278722610339132e-02, 9.566402403655620e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_xc_hcth_407_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_hcth_407", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [1.021375033664179e-01, 0.000000000000000e+00, -7.220479855161298e+20, 3.966348061236097e-03, 0.000000000000000e+00, -4.957773911032857e+20, -8.428078562608074e-02, 0.000000000000000e+00, -1.924650575314499e+20, -1.878367330539667e+01, 0.000000000000000e+00, 8.005132502133870e+19, -5.901592493605343e+01, 0.000000000000000e+00, 4.706627120739602e+14]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
