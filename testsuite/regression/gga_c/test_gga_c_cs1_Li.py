
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_cs1_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_cs1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-5.014824293955897e-02, -4.629220440120434e-02, -2.024194100705756e-03, -6.116507047468900e-03, 1.680896297866659e-03, 3.011777301041895e-03, 6.619951194421702e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_cs1_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_cs1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-5.455286131471203e-02, -5.436251710042202e-02, -5.573133517944984e-02, -5.555759370215654e-02, -5.949506369906379e-02, -5.947744845847969e-02, -7.758202594282910e-03, -4.599278008790979e-02, -1.207391743015411e-02, -2.514087283122250e-02, 4.011695913533000e-03, 3.901281852111236e-03, 1.128752343153146e-04, 1.971740147922522e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_cs1_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_cs1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([3.426400890440848e-06, 1.903559290333688e-08, 3.406355707328666e-06, 3.401984756286038e-05, 2.025506712836066e-07, 3.385483790737835e-05, 2.771768721624228e-02, 3.406853726718826e-04, 2.768096863129638e-02, 1.362932821014516e-01, 4.671424923261951e-06, 2.391959572238338e-01, 3.112767863841341e+01, 5.433709604084945e-07, 1.580079678293951e+00, 2.456300745407244e-01, 5.944895543815402e-03, 2.296316531388017e-01, 1.163208281667670e+00, 2.512559379948644e-02, 1.659931235027038e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
