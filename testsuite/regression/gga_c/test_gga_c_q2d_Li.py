
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_q2d_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_q2d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-5.998896163030750e-02, -4.346072137300211e-02, -3.082490383005075e-03, -1.516244912637255e-02, -1.348097743281305e-03, -3.325144961671289e-04, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_q2d_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_q2d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.160116456357027e-01, -1.158858173627468e-01, -1.014547547553956e-01, -1.013602645236172e-01, -1.590657789395799e-02, -1.591223576867854e-02, -2.421895783812505e-02, -9.764053534162104e-02, -6.581814009229584e-03, 2.681030010545644e-01, 4.721503866541696e-04, 4.697225985712559e-04, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_q2d_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_q2d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([4.595408903458736e-05, 9.190817806917472e-05, 4.595408903458736e-05, 1.490440300700200e-04, 2.980880601400400e-04, 1.490440300700200e-04, 3.118186053146789e-03, 6.236372106293579e-03, 3.118186053146789e-03, 3.381134496822776e+00, 6.762268993645552e+00, 3.381134496822776e+00, 1.129313011983172e+01, 2.258626023966345e+01, 1.129313011983172e+01, -4.443711191617074e+00, -8.887422383234147e+00, -4.443711191617074e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
