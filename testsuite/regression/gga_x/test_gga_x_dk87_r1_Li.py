
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_dk87_r1_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_dk87_r1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.797791258501034e+00, -1.291725991462863e+00, -4.182955481950157e-01, -1.600140791410422e-01, -8.048939124708088e-02, -1.396102528499172e-01, -8.496807562136791e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_dk87_r1_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_dk87_r1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.221093149991300e+00, -2.223226121783651e+00, -1.506983722866130e+00, -1.508319105868035e+00, -3.704218558334693e-01, -3.703065725547368e-01, -2.040580667242266e-01, -1.492327326447579e-02, -7.792795793958239e-02, -4.745920897120625e-04, -1.569030143411156e-02, -1.557733327898583e-02, -3.170177351680053e-04, -2.253751064986825e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_dk87_r1_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_dk87_r1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-2.998130502987344e-04, 0.000000000000000e+00, -2.987823839406995e-04, -1.130829670376322e-03, 0.000000000000000e+00, -1.127372039577483e-03, -9.082557148040440e-02, 0.000000000000000e+00, -9.078560969422643e-02, -4.526951939232258e+00, 0.000000000000000e+00, -1.630449082532228e+03, -6.431485504553036e+01, 0.000000000000000e+00, -8.121965387339574e+07, -1.407922225035745e+03, 0.000000000000000e+00, -1.414614822136770e+03, -2.517506669295925e+08, 0.000000000000000e+00, -7.664241297238337e+08])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
