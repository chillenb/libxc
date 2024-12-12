
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_xc_mpw1b95_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_mpw1b95", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.289354553160868e+00, -9.294824635182010e-01, -3.043216899658682e-01, -1.109027572851677e-01, -5.590666878293452e-02, -1.194610839109312e-03, -1.172257067378601e-06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_xc_mpw1b95_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_mpw1b95", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.609937859328042e+00, -1.611197897635443e+00, -1.113006074039089e+00, -1.113752899757603e+00, -2.764244777658853e-01, -2.763338112722635e-01, -1.434102484657008e-01, -4.656158516080719e-03, -5.260208458278454e-02, -1.859035714966966e-06, -4.567155292995314e-03, -4.251972179537887e-03, -7.825409390189270e-06, -8.868377277709695e-08])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_mpw1b95_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_mpw1b95", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.491580712578006e-04, 0.000000000000000e+00, -1.487182027364827e-04, -5.535556092306597e-04, 0.000000000000000e+00, -5.521409115611365e-04, -6.063319622614918e-02, 0.000000000000000e+00, -6.065036025274956e-02, -5.719003200175394e-01, 0.000000000000000e+00, 2.925430368400773e+01, -4.393459510317179e+01, 0.000000000000000e+00, 9.813318716470668e+02, 2.463161599384900e+01, 0.000000000000000e+00, 2.231795988835737e+01, 1.805362572041166e+04, 0.000000000000000e+00, 4.070643524784727e+02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_mpw1b95_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_mpw1b95", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-1.535420278481735e-03, -1.534572015769308e-03, -1.754667312298453e-03, -1.753913243294452e-03, -4.414736987621208e-04, -4.386980204772437e-04, -9.437213115668527e-02, -1.176431803685671e-07, -1.559459506086730e-02, -3.855053562272995e-11, -1.373116706943418e-07, -1.256638198788993e-07, -8.551510937470655e-12, -4.468127138692301e-12])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
