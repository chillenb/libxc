
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_airy_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_airy", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.726814849516964e+00, -1.204274458695834e+00, -3.611644082571241e-01, -1.562690599563058e-01, -7.016847222178098e-02, -8.798856248300019e-02, -3.624995661191417e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_airy_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_airy", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.303480842200968e+00, -2.305648204206340e+00, -1.550792210034463e+00, -1.552235719844699e+00, -3.228936873425132e-01, -3.227029288752395e-01, -2.096595412258381e-01, -2.403136457807762e-02, -7.225091404186824e-02, -5.183364181346478e-03, -2.481132011152452e-02, -2.483287476635731e-02, -5.221568689012719e-03, -4.615172242862025e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_airy_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_airy", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([3.598950560617516e-06, 0.000000000000000e+00, 3.746203994690070e-06, -2.857488079266430e-04, 0.000000000000000e+00, -2.842962706841590e-04, -7.693554270356900e-02, 0.000000000000000e+00, -7.694494066655286e-02, 6.179540608968634e-01, 0.000000000000000e+00, -8.811188444780992e+02, -4.662480063424796e+01, 0.000000000000000e+00, -3.285792021962761e+07, -7.645727143870110e+02, 0.000000000000000e+00, -7.665304085398606e+02, -9.700844207242632e+07, 0.000000000000000e+00, -2.879202979500613e+08])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
