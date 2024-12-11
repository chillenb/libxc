
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_mvsb_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mvsb", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-2.178898063299192e+00, -1.433106351909674e+00, -2.854170856997563e-01, -1.904715062755123e-01, -6.044963444737322e-02, -2.025550296220105e-03, -4.369556749394064e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_mvsb_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mvsb", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.588495856674344e+00, -2.590414811327204e+00, -2.353008607118316e+00, -2.357250392921372e+00, -3.742503042991869e-01, -3.738092453016924e-01, -2.395824716505078e-01, -5.134101828955713e-03, -7.772673808798629e-02, -3.655839044042635e-05, -2.097895765112826e-03, -5.440379054354206e-03, -1.017461541350270e-06, -2.183783667931084e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mvsb_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mvsb", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([6.358494023353550e-04, 0.000000000000000e+00, 6.344573485785802e-04, -1.728004009389718e-03, 0.000000000000000e+00, -1.736698503479369e-03, -3.158363724744281e-02, 0.000000000000000e+00, -3.227402788346397e-02, 8.975222196130956e+00, 0.000000000000000e+00, 7.225919899845833e+00, -2.531305252662384e+01, 0.000000000000000e+00, 4.075391665872447e+03, 5.236555288304663e-02, 0.000000000000000e+00, 6.557017331256243e+00, 2.571415445732135e-03, 0.000000000000000e+00, -3.247994107373919e+05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mvsb_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mvsb", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-4.494481360491737e-02, -4.494685172068932e-02, 6.482960507574116e-02, 6.513405077934975e-02, -1.171789650780432e-03, -1.230023453282719e-03, -4.114582845204592e-01, 1.344453334218985e-04, -2.405376639627488e-02, 2.417742381014043e-06, 1.132381472700379e-06, 1.387952066021707e-04, 4.546002882282868e-13, -1.687484795652016e-12])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
