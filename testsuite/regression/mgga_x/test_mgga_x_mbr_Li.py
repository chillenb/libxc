
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_mbr_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mbr", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.752865165094631e+00, -1.276457690260470e+00, -4.361819299077551e-01, -1.551022829372977e-01, -8.319431573249814e-02, -2.200898973301076e-01, -3.417561660809738e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_mbr_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mbr", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.113327417673147e+00, -2.115394299121999e+00, -1.420305670024707e+00, -1.421587981354858e+00, -3.946447673461772e-01, -3.941178166962541e-01, -1.945084924896497e-01, -4.821059165191918e-02, -8.020254829203674e-02, -1.263142957359424e-02, -2.053742345974850e-01, -4.944379318260446e-02, -2.617423040115939e+00, -1.076233245845757e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mbr_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mbr", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-3.491568958447537e-04, 0.000000000000000e+00, -3.479678205096188e-04, -1.326457065555360e-03, 0.000000000000000e+00, -1.322415367684915e-03, -7.567150333294502e-02, 0.000000000000000e+00, -7.587964036892945e-02, -5.527684827677933e+00, 0.000000000000000e+00, -7.708923692798138e+02, -5.716546518248507e+01, 0.000000000000000e+00, -2.522969476209688e+07, -1.428142900988965e+02, 0.000000000000000e+00, -6.721928604484855e+02, -2.437124254877959e+05, 0.000000000000000e+00, -2.190636925260546e+08])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mbr_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mbr", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-2.719746407536157e-03, -2.716303248032382e-03, -4.274087797195891e-03, -4.270498666432404e-03, -5.668359408618221e-03, -5.679304809709015e-03, -2.713207061212089e-02, -3.940679862211121e-03, -3.850881826714468e-02, -4.140719724416635e-03, -8.485341955644593e-04, -3.908127788757420e-03, -1.192048162456247e-05, -3.849811845129638e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
