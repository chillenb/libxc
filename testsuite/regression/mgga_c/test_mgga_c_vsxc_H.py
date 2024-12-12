
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_vsxc_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_vsxc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.065926955207835e-10, -2.291320028335873e-11, -9.738004929946874e-12, -2.804980215865283e-14, -5.177054388933450e-13])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_vsxc_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_vsxc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-6.927386164420456e-02, -1.780516748108563e-01, -4.289078983295840e-02, -1.699594528100074e-01, -2.729603264156907e-02, -1.323220428811156e-01, -6.570157130826616e-02, -6.073324485636818e-02, -3.154860777585695e-05, 1.155769378659515e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_vsxc_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_vsxc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([2.833522096275178e+00, 0.000000000000000e+00, 1.245866268931512e+19, 5.312623890768325e-02, 0.000000000000000e+00, 1.190066056853395e+19, 1.589672595909489e-01, 0.000000000000000e+00, 9.270490408597463e+18, 1.972097549597789e+01, 0.000000000000000e+00, 4.277897205930526e+18, 6.722614227448486e+01, 0.000000000000000e+00, 6.151522872723476e+15])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_vsxc_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_vsxc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-6.757906564867495e+00, 6.645380975032918e+03, -9.133937499033785e-02, 6.346745247794562e+03, -5.468807599245547e-02, 4.942376824677165e+03, -1.292192378438838e-01, 2.265570873480293e+03, -4.605645502836044e-05, -9.576225390518292e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
