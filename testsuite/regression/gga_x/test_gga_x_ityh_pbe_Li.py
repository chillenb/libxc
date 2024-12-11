
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_ityh_pbe_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_ityh_pbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.606392083214557e+00, -1.090659686561520e+00, -1.806078753153254e-01, -4.354818483914281e-02, -4.020308713599879e-03, -2.644418243290553e-05, -1.818419751095592e-10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_ityh_pbe_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_ityh_pbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.075161288924826e+00, -2.077268466877665e+00, -1.364563703548025e+00, -1.365905277140649e+00, -2.549949016706999e-01, -2.549569680647530e-01, -7.558439863397375e-02, -4.595818892210082e-05, -7.754144028623890e-03, -1.469230238827698e-09, -5.342909809697457e-05, -5.227354997273580e-05, -4.378322115425854e-10, -1.573074474240525e-10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_ityh_pbe_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_ityh_pbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-2.155773170004479e-04, 0.000000000000000e+00, -2.148694014882237e-04, -7.871317063018578e-04, 0.000000000000000e+00, -7.847866611947362e-04, -1.840574508813905e-02, 0.000000000000000e+00, -1.833995712803793e-02, -3.969990505566554e-01, 0.000000000000000e+00, -5.155109405846020e-07, -2.292212975387168e-01, 0.000000000000000e+00, -3.342718848969111e-12, -6.404549069077374e-07, 0.000000000000000e+00, -5.808554949651422e-07, -4.843570906958170e-13, 0.000000000000000e+00, -1.770847111931286e-13])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
