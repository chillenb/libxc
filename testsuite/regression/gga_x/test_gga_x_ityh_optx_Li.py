
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_ityh_optx_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_ityh_optx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-3.967756803109901e+00, -2.652293089856148e+00, -3.821988295307177e-01, -1.139664268576974e-01, -1.068279758218167e-02, -7.188245085305396e-05, -4.950644795794764e-10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_ityh_optx_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_ityh_optx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-5.230527879931643e+00, -5.235069756895559e+00, -3.607590257712104e+00, -3.610898511647862e+00, -5.988569923551561e-01, -5.984657358715240e-01, -1.941822175676734e-01, -1.248853405468680e-04, -2.071870681060199e-02, -3.999971706892925e-09, -1.451576994621207e-04, -1.420225352958896e-04, -1.191997183069890e-09, -4.282693416971015e-10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_ityh_optx_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_ityh_optx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-3.395020789651502e-04, 0.000000000000000e+00, -3.395368184480035e-04, -3.423132688581958e-04, 0.000000000000000e+00, -3.420126168510259e-04, -2.389615286816545e-04, 0.000000000000000e+00, -2.374149462877191e-04, -1.932535346394680e+00, 0.000000000000000e+00, -5.305175454434265e-09, -8.669644342823532e-03, 0.000000000000000e+00, -3.444578884898802e-14, -6.590151612093911e-09, 0.000000000000000e+00, -5.976705393784156e-09, -4.991162464345787e-15, 0.000000000000000e+00, -1.824810644339847e-15])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
