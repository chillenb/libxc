
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_hcth_a_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_hcth_a", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-7.313753840600310e-04, -8.266167974769112e-03, -1.281357436262499e-02, -9.918481644473218e-03, -1.203429419462027e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_hcth_a_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_hcth_a", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.302550432824796e-04, -2.241451564947146e-01, 7.972764047733089e-03, -2.267959700105574e-01, -1.790714284861547e-03, -1.932164604348658e-01, -1.090170368231281e-02, 3.185195991048032e-04, -1.537590985154598e-03, 3.326897317214947e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_hcth_a_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_hcth_a", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.059578309950671e-02, 0.000000000000000e+00, -1.351845008282225e+22, -8.024507275460290e-03, 0.000000000000000e+00, -1.099469048962861e+22, -2.800295260747717e-02, 0.000000000000000e+00, -5.450757083265644e+21, -9.363963665791908e-02, 0.000000000000000e+00, 3.039315585153033e+21, -1.347103160783178e-01, 0.000000000000000e+00, 5.101728692435259e+15])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
