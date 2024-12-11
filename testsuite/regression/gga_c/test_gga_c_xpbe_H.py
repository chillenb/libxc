
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_xpbe_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_xpbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-3.193012953950008e-02, -1.772711054923840e-02, -7.974805437477858e-03, -1.238096106414652e-04, -1.322633677927620e-10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_xpbe_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_xpbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-3.713535958486652e-02, 2.269793086533077e+00, -3.845437215886227e-02, 9.416986605641701e+01, -2.599101249714152e-02, 5.568175955038665e+01, -7.475789000096246e-04, 3.602145163063459e-01, -8.616413639441628e-10, 6.381615633285417e-09])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_xpbe_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_xpbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([2.154483683433348e-02, 4.308967366866696e-02, 2.154483683433348e-02, 8.977937198013505e-03, 1.795587439602701e-02, 8.977937198013505e-03, 3.785843600355621e-02, 7.571687200711241e-02, 3.785843600355621e-02, 6.906665483776245e-02, 1.381333096755255e-01, 6.906665483776245e-02, 5.635950587767583e-04, 1.127190116839631e-03, 5.635950587767583e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
