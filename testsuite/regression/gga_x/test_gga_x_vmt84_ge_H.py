
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_vmt84_ge_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_vmt84_ge", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-6.218547938690901e-01, -5.689862409035330e-01, -3.462179284397418e-01, -1.292297804287424e-01, -3.680737304729215e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_vmt84_ge_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_vmt84_ge", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-8.285212489442504e-01, -5.423381180197838e-17, -7.283328234274129e-01, -1.667159499834881e-16, -4.110748331245689e-01, 5.430794769865400e-17, -1.149257118616122e-01, -6.681698831706724e-17, -5.940452133520433e-03, -5.921144277162941e-19]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_vmt84_ge_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_vmt84_ge", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-9.486619383147177e-03, 0.000000000000000e+00, 0.000000000000000e+00, -1.408124318715883e-02, 0.000000000000000e+00, 0.000000000000000e+00, -1.103958147388702e-01, 0.000000000000000e+00, 0.000000000000000e+00, -6.458764650310082e+00, 0.000000000000000e+00, 0.000000000000000e+00, 8.252898632716405e+02, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
