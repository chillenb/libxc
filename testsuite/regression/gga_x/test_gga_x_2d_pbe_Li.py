
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_2d_pbe_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_2d_pbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-3.964379239588228e+00, -2.409839282644303e+00, -3.725311151227765e-01, -1.204434073576776e-01, -3.759441578131593e-02, -2.977523738521507e-03, -7.653356888625140e-06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_2d_pbe_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_2d_pbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-5.580998261261304e+00, -5.588890409341920e+00, -3.145439677900281e+00, -3.149574098605210e+00, -5.351589135865156e-01, -5.349547953132379e-01, -1.484697683092805e-01, -4.164227811359684e-03, -5.524793231262360e-02, -2.352021423553621e-05, -4.490453627945618e-03, -4.441554814416599e-03, -1.283955415689483e-05, -7.696098700909219e-06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_2d_pbe_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_2d_pbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-5.510866036221406e-04, 0.000000000000000e+00, -5.490781400676443e-04, -2.188093164425032e-03, 0.000000000000000e+00, -2.181389107442327e-03, -1.025776178261965e-02, 0.000000000000000e+00, -1.019643106065253e-02, -1.395043760248587e+01, 0.000000000000000e+00, -9.550954409410295e-05, -2.224607987257254e+00, 0.000000000000000e+00, -3.442160156735657e-06, -1.046913513564949e-04, 0.000000000000000e+00, -9.668591929901648e-05, -1.367878203932799e-06, 0.000000000000000e+00, -1.173620913134555e-06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
