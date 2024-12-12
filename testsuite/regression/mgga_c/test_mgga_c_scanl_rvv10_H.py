
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_scanl_rvv10_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_scanl_rvv10", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-7.424616477180734e-16, -9.714451465470120e-16, -5.519584462595033e-04, -4.667745747487701e-03, -2.911020214038745e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_scanl_rvv10_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_scanl_rvv10", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-7.489301265344016e-17, -2.199798185667858e-01, -9.210777006059275e-17, -2.101331250970284e-01, -1.418124245873291e-02, 1.171550144000164e+00, -5.561609384491019e-02, -2.677389264971583e+00, -5.566147936945416e-04, 1.283888786293379e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_scanl_rvv10_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_scanl_rvv10", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([1.004972254232514e-17, 8.467745752978754e-03, 4.233872876489377e-03, 3.789919438003834e-17, 9.772875833021489e-03, 4.886437916510744e-03, 3.051336349565278e-02, 1.046722004570729e-01, 5.233610022853647e-02, 1.470819332207355e+01, 2.546938337650606e+01, 1.273469168825303e+01, 1.511989167977952e+02, 3.743876400859436e+02, 1.871938200429718e+02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_scanl_rvv10_H_2_vlapl():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_scanl_rvv10", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = numpy.asarray([0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, -1.904660565526701e-03, 0.000000000000000e+00, -1.804626523583903e-02, 0.000000000000000e+00, 4.583566177924238e-07, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
