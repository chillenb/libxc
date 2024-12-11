
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_ktbm_8_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_8", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.936383418426626e+00, -1.287109490479308e+00, -2.527114358936241e-01, -1.779427845832335e-01, -5.470106558788419e-02, -1.122078489934749e-02, -2.060470462395835e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_ktbm_8_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_8", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.720002578587867e+00, -2.722595695043241e+00, -1.895300261925909e+00, -1.896934263368815e+00, -3.318401616431015e-01, -3.319003768171499e-01, -2.451036419704208e-01, -1.288407596726991e-02, -7.670639830587438e-02, -4.085827653453006e-04, -1.354715971310238e-02, -1.345002560169517e-02, -2.729097469589526e-04, -1.975305775924576e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_8_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_8", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-5.483987045990894e-04, 0.000000000000000e+00, -5.465129678172385e-04, -2.093176561613260e-03, 0.000000000000000e+00, -2.087396228556065e-03, -4.857628742686623e-02, 0.000000000000000e+00, -5.057336201421176e-02, -8.527394535884977e+00, 0.000000000000000e+00, -3.577994072713658e+01, -7.037926206097319e+01, 0.000000000000000e+00, -8.973970311816195e+04, -6.662621781371828e-01, 0.000000000000000e+00, -3.198876769915625e+01, -1.358920290058016e+00, 0.000000000000000e+00, -9.595536438608944e+04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_8_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_8", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([1.987754835371984e-02, 1.985705788821085e-02, 2.632390808103200e-02, 2.631768873762862e-02, -9.500231590508291e-04, -9.179116556975857e-04, 2.211205272149160e-01, 4.573051837164148e-04, 3.157975589085113e-02, 3.656320121543765e-05, 9.894940322731832e-06, 4.651414712842967e-04, 1.649947397819111e-10, -6.549009383163791e-12])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
