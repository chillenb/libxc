
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_b88_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_b88", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.802451127976592e+00, -1.290833460011208e+00, -4.322034088033075e-01, -1.605897927949022e-01, -8.126244578099942e-02, -1.332056719320396e-01, -5.361399227810711e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_b88_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_b88", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.242378622506531e+00, -2.244498551653618e+00, -1.524830129717005e+00, -1.526190468882155e+00, -3.397388066384502e-01, -3.396131906024259e-01, -2.050706435177237e-01, -3.582062777896466e-02, -7.334672068946504e-02, -7.724254008484471e-03, -3.666535985119013e-02, -3.684142807190940e-02, -7.461236846304174e-03, -6.453098143330768e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_b88_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_b88", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-2.739685323692003e-04, 0.000000000000000e+00, -2.730623591257954e-04, -1.030512736727857e-03, 0.000000000000000e+00, -1.027285998405760e-03, -1.146827747476768e-01, 0.000000000000000e+00, -1.146490279010521e-01, -4.409403314773193e+00, 0.000000000000000e+00, -1.339589798703576e+03, -7.659518022198580e+01, 0.000000000000000e+00, -4.850303302271806e+07, -1.164866909031208e+03, 0.000000000000000e+00, -1.166725787563364e+03, -1.440002236570638e+08, 0.000000000000000e+00, -4.289611223828076e+08])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
