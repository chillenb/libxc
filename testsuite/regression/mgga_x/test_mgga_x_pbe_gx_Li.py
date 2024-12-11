
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_pbe_gx_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_pbe_gx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-2.050084083960583e+00, -1.356759460930664e+00, -2.567524314635121e-01, -1.884509744670584e-01, -5.708775059523367e-02, -4.106761489219093e-05, -3.688423192687207e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_pbe_gx_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_pbe_gx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.884456479256919e+00, -2.887114890579094e+00, -2.036799974557977e+00, -2.038299680965265e+00, -3.571022557482202e-01, -3.575197128143131e-01, -2.596682341478578e-01, 5.593281331022329e-02, -8.140822723397724e-02, 1.792957301474547e-03, 5.943510096014453e-02, 5.835094683694302e-02, 1.197627758584658e-03, -1.860583909431408e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_pbe_gx_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_pbe_gx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-3.258984493327862e-04, 0.000000000000000e+00, -3.247102165815073e-04, -1.547571421275478e-03, 0.000000000000000e+00, -1.541212775515427e-03, 7.832922946649240e-02, 0.000000000000000e+00, 7.804746524886538e-02, -4.761212401577544e+00, 0.000000000000000e+00, -1.438178141888416e+03, 2.485397670115562e+01, 0.000000000000000e+00, -3.636206128416383e+06, -2.699217865036953e+01, 0.000000000000000e+00, -1.285154557990800e+03, -5.506399711354478e+01, 0.000000000000000e+00, 8.388781566404744e+11])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_pbe_gx_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_pbe_gx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([2.169710521055146e-02, 2.167705367572001e-02, 3.344650858479086e-02, 3.340055085459106e-02, 2.695692524975795e-03, 2.815046749766149e-03, 2.366053314583921e-01, 1.839183288016973e-02, 4.430316555580477e-02, 1.481524060209680e-03, 4.008774379519238e-04, 1.869836983336668e-02, 6.685653265765240e-09, 4.870078895627024e-12])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
