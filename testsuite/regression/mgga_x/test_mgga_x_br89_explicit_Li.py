
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_br89_explicit_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_br89_explicit", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.382820109865281e+00, -1.207219819808327e+00, -4.834082114637349e-01, -1.469858898755506e-01, -8.663521731957678e-02, -2.748592977476321e-01, -1.907535500975168e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_br89_explicit_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_br89_explicit", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.870105475266117e+00, -1.871878320358321e+00, -1.753296076344643e+00, -1.754905270613644e+00, -4.232460088234682e-01, -4.227546040636192e-01, -2.240620127314839e-01, -7.256576645266252e-02, -8.457650497490435e-02, -1.607860264534786e-02, -2.351270757597363e-01, -7.210614463652722e-02, -1.692908792678262e+00, -1.015637364853249e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_br89_explicit_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_br89_explicit", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-6.181934467213427e-07, 0.000000000000000e+00, -6.181773168185704e-07, -9.884038661595799e-04, 0.000000000000000e+00, -9.856421993325436e-04, -7.962962569963553e-02, 0.000000000000000e+00, -7.980238099954859e-02, -2.064043515898824e+00, 0.000000000000000e+00, -8.455340307639534e+02, -6.382502597318933e+01, 0.000000000000000e+00, -3.112928104319529e+07, -2.401870599807354e+02, 0.000000000000000e+00, -7.488104875052054e+02, -1.927862937769328e+05, 0.000000000000000e+00, -2.775265776932566e+08])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_br89_explicit_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_br89_explicit", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = numpy.asarray([-1.002418697754543e-05, -1.005104532871904e-05, -5.363978364576818e-03, -5.362660941506672e-03, -5.991288505554718e-03, -5.996629414932751e-03, -2.476028993762273e-02, -3.374658281257540e-03, -4.769875805470480e-02, -3.963491101600474e-03, -1.114706135682781e-03, -3.399946632728156e-03, -7.314797225364612e-06, -3.783311391096099e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_br89_explicit_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_br89_explicit", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([3.207739832814614e-05, 3.216334505190049e-05, 1.716473076664581e-02, 1.716051501282138e-02, 1.917212321777511e-02, 1.918921412778482e-02, 7.923292780039262e-02, 1.079890650002412e-02, 1.526360257750548e-01, 1.268317152512152e-02, 3.567059634184896e-03, 1.087982922473010e-02, 2.340735112116676e-05, 1.210659645150752e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
