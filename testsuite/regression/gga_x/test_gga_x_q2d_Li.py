
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_q2d_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_q2d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.769412201177607e+00, -1.250406222959321e+00, -2.268352043313879e-01, -1.586710273346666e-01, -6.844436081805737e-02, -8.169031014351682e-04, -2.593153877059012e-06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_q2d_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_q2d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.272178335065488e+00, -2.274293486581005e+00, -1.554934702834924e+00, -1.556301850728849e+00, -8.776058978189588e-01, -8.774158350032849e-01, -2.069804682538378e-01, -1.514849477960172e-03, -1.108314860714479e-01, -1.096512878712047e-05, -1.634585259864093e-03, -1.604009724789900e-03, -5.753460552520269e-06, -3.606887586431603e-06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_q2d_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_q2d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.473925971910303e-04, 0.000000000000000e+00, -1.468805961797810e-04, -5.879249019268770e-04, 0.000000000000000e+00, -5.860418610476008e-04, 2.786903766222312e-01, 0.000000000000000e+00, 2.787193695707363e-01, -2.263068077748610e+00, 0.000000000000000e+00, 4.769208253511068e+00, 4.282879908884295e+01, 0.000000000000000e+00, 2.779162664095426e+03, 4.455594580485627e+00, 0.000000000000000e+00, 4.321206657257133e+00, 4.175505686437517e+03, 0.000000000000000e+00, 8.716935617316960e+03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
