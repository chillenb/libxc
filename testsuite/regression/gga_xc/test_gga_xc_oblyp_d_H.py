
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_xc_oblyp_d_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_oblyp_d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-6.221129311959495e-01, -5.792981926889832e-01, -3.602388019165267e-01, -1.461121566940916e-01, -5.947161007012587e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_xc_oblyp_d_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_oblyp_d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-8.281822745355694e-01, -2.515898863879487e-01, -7.195830079689985e-01, -2.688963089662590e-01, -4.051098565078508e-01, -2.167335723037147e-01, -1.032314139336100e-01, -4.235490467398996e-02, -1.512609833492756e-02, -2.421952677946944e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_xc_oblyp_d_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_oblyp_d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.996536558982113e-02, 3.302700435410080e-02, 2.476076625267766e-02, -2.453184118624402e-02, 5.272675688452872e-02, 3.949374514035636e-02, -1.642504694179684e-01, 4.683678247165274e-01, 3.512654903397595e-01, -1.030878026380738e+01, 2.575493420287057e+01, 1.931616797079035e+01, -5.127635652265023e+04, 1.567218525933866e-14, 1.175412398789372e-14])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
