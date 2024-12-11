
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_mb88_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_mb88", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.755306093871720e+00, -1.231540332464035e+00, -3.555500524148386e-01, -1.579612827294209e-01, -6.986550171173504e-02, -1.284568327342921e-01, -5.357356574598113e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_mb88_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_mb88", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.289675694376820e+00, -2.291779842902239e+00, -1.574590022932874e+00, -1.575963793948967e+00, -3.287170695962117e-01, -3.284481499180308e-01, -2.079564911367036e-01, -2.849910288623921e-02, -7.530667229213829e-02, -7.607436462130795e-03, -2.888685502534034e-02, -2.915474974814749e-02, -7.390856939611347e-03, -6.405598883633544e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_mb88_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_mb88", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-8.516558728087413e-05, 0.000000000000000e+00, -8.486653433582102e-05, -3.518393671584079e-04, 0.000000000000000e+00, -3.506734514179657e-04, -7.048513124302455e-02, 0.000000000000000e+00, -7.053259742941707e-02, -1.300290430541249e+00, 0.000000000000000e+00, -1.352412643830311e+03, -3.905412138386232e+01, 0.000000000000000e+00, -4.851802475499532e+07, -1.176583193901106e+03, 0.000000000000000e+00, -1.178204324429939e+03, -1.440251606300814e+08, 0.000000000000000e+00, -4.290161129445583e+08])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
