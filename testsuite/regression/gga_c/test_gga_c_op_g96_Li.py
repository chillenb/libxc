
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_op_g96_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_op_g96", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-5.352084493639449e-02, -4.635721575784840e-02, -1.173587712659869e-02, -4.351177682762987e-06, -4.924158875277586e-10, -1.804359080553705e-07, -1.673057496149299e-14])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_op_g96_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_op_g96", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-6.890553005313894e-02, -6.873533990186610e-02, -6.718693660977554e-02, -6.702146454573389e-02, -3.390086124464489e-02, -3.397720861238894e-02, -1.935664731696112e-06, -2.227151708032216e-02, -6.305758971434830e-10, -3.303735912993994e-03, -1.153671836754466e-06, -1.156623899086828e-06, -7.870429127688322e-14, -2.135408425448671e-13])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_op_g96_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_op_g96", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([2.163528791025300e-05, 0.000000000000000e+00, 2.147283950181035e-05, 9.310420132223119e-05, 0.000000000000000e+00, 9.243216493378710e-05, 9.927682535582886e-03, 0.000000000000000e+00, 9.936741964636920e-03, 2.367347700800000e-04, 0.000000000000000e+00, 1.062194381873197e+02, 1.096401428592909e-06, 0.000000000000000e+00, 3.773409551031964e+05, 7.986991089414645e-03, 0.000000000000000e+00, 7.838631776168614e-03, 1.462272894844524e-04, 0.000000000000000e+00, 1.307266918545144e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
