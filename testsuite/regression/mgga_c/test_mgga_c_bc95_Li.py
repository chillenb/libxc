
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_bc95_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_bc95", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-4.517333869970402e-02, -3.832557383986378e-02, -9.706680231042746e-03, -9.569134312279086e-05, -2.182532448212174e-10, -1.327870368225530e-05, -1.126992763648026e-06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_bc95_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_bc95", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-6.288684052448007e-02, -6.268482273895758e-02, -6.060251490315760e-02, -6.041181071667955e-02, -2.867638272992059e-02, -2.859660787978142e-02, -1.936693277684212e-03, -8.034624805272761e-04, -1.117496387146263e-03, -1.198429609781657e-06, -1.106973661166908e-04, -1.336876614042987e-05, -7.633703072117562e-06, -3.639877476854735e-10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_bc95_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_bc95", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([4.132575779267206e-05, 0.000000000000000e+00, 4.113737666855524e-05, 1.594551976290499e-04, 0.000000000000000e+00, 1.586449484834225e-04, 9.709021590385846e-03, 0.000000000000000e+00, 9.630978071103387e-03, 2.502270307487512e+00, 0.000000000000000e+00, 5.648708779456898e+00, 6.520909806496228e+00, 0.000000000000000e+00, 6.632484764765807e+02, 9.633536558453863e-01, 0.000000000000000e+00, 8.121669096347492e-02, 1.778945379389083e+04, 0.000000000000000e+00, 1.857317276782201e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_bc95_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_bc95", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-1.535420278481735e-03, -1.534572015769308e-03, -1.754667312298453e-03, -1.753913243294452e-03, -4.414736987621208e-04, -4.386980204772437e-04, -9.437213115668527e-02, -1.176431803685671e-07, -1.559459506086730e-02, -3.855053562272995e-11, -1.373116706943418e-07, -1.256638198788993e-07, -8.551510937470655e-12, -4.468127138692301e-12])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
