
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_pbe_erf_gws_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_pbe_erf_gws", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-2.283412555957990e-02, -1.535597754783565e-02, -7.882089961704227e-03, -2.476668289717114e-04, -9.720218380193657e-11])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_pbe_erf_gws_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_pbe_erf_gws", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.878434327107222e-02, 1.143567132690991e+02, -2.992015482760224e-02, 9.717057788698861e+01, -1.756185041376101e-02, 2.694231733444943e+01, -5.895107920280748e-04, 1.582507919087009e-02, -2.572485880997927e-10, -1.076149053549479e-06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_pbe_erf_gws_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_pbe_erf_gws", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([6.398875893005251e-03, 1.279775178601050e-02, 6.398875893005251e-03, 5.739303854663349e-03, 1.147860770932670e-02, 5.739303854663349e-03, 1.773815760814102e-02, 3.547631521628205e-02, 1.773815760814102e-02, 7.021976231206931e-04, 1.404395246241386e-03, 7.021976231206931e-04, 8.146237058621938e-14, 1.629247411724387e-13, 8.146237058621938e-14])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
