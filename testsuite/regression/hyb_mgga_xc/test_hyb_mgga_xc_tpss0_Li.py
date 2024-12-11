
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_xc_tpss0_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_tpss0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.406606491195023e+00, -9.908724258718355e-01, -2.729078552940851e-01, -1.370212703724245e-01, -5.855002329510586e-02, -1.541308863355283e-02, -2.629231168491740e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_xc_tpss0_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_tpss0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.864826732396466e+00, -1.866264639153552e+00, -1.313690262372222e+00, -1.314616767546297e+00, -3.518571522251552e-01, -3.516382312552860e-01, -1.803501279815038e-01, -1.438976554425545e-01, -7.516497903128400e-02, -7.214965356237907e-02, -2.063029279179982e-02, -2.044492217192749e-02, -4.156173146338845e-04, -1.695022775903474e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_tpss0_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_tpss0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.189820044868792e-04, 2.772505973492434e-04, -1.169835055957481e-04, -1.244083773491391e-04, 1.193829360748139e-03, -1.229027500750196e-04, 1.498417310840712e-01, 3.593578553367118e-01, 1.493084377202372e-01, -1.423509302918334e+00, 8.341737401120239e+00, 3.962331395374586e+00, 1.437450722868072e+02, 3.363362362756724e+02, 1.668357445136749e+02, -8.874791074968291e-05, 1.079340474789106e-08, -1.979093781843348e-01, -6.081268833118538e-11, -1.923003370021578e-15, -1.168048397881161e+11])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_tpss0_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_tpss0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([1.742757374610352e-03, 1.728611578209718e-03, 1.773108882597880e-03, 1.776262323974694e-03, -4.766653772229204e-04, -5.044565639379200e-04, 2.012472674750179e-02, -3.293551529024618e-10, -1.167457754155709e-02, 2.607807173520796e-17, 5.681378563954287e-16, 5.772785063278045e-11, 1.069178041425852e-33, -4.121311491174611e-13])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
