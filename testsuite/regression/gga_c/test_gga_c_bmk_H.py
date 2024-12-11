
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_bmk_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_bmk", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([4.541554589979137e-02, -2.121253475623574e-02, 1.751550440616274e-03, 7.801651371847105e-04, 4.589348644132795e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_bmk_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_bmk", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([1.126658841382079e-01, -3.252669622808604e-01, -1.158606374957714e-01, -2.856858948875026e-01, -1.367438203246169e-02, -1.941362696298502e-01, 1.869174564033640e-03, -3.749171130687481e-02, 5.874143304024824e-05, 2.360190585370376e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_bmk_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_bmk", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-9.462297892289191e-01, 0.000000000000000e+00, 5.710389266020014e+20, 4.272840493491249e-02, 0.000000000000000e+00, 4.277697101426795e+20, 3.422641671723362e-02, 0.000000000000000e+00, 1.980279292894868e+20, -1.064077499422998e-01, 0.000000000000000e+00, 3.188837703422808e+19, -1.794543435823830e-01, 0.000000000000000e+00, 1.072144644906816e+14])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
