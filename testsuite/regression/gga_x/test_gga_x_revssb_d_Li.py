
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_revssb_d_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_revssb_d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.887625499846360e+00, -1.341834901975175e+00, -4.304992304220207e-01, -1.683582890406738e-01, -8.139789028453875e-02, -2.054512430528483e-02, -3.838585467975121e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_revssb_d_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_revssb_d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.446492648611130e+00, -2.448966325451823e+00, -1.560578964339051e+00, -1.562237646112697e+00, -4.097258317393199e-01, -4.098811081208936e-01, -2.274419034076523e-01, -2.611531237463836e-02, -7.832640298379370e-02, -8.296429588359809e-04, -2.745674817514866e-02, -2.725947951057180e-02, -5.541553011545962e-04, -3.939540422767455e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_revssb_d_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_revssb_d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.192119101822961e-04, 0.000000000000000e+00, -1.184304971159302e-04, -1.210258666765010e-03, 0.000000000000000e+00, -1.205110334752617e-03, -8.670872848153410e-02, 0.000000000000000e+00, -8.653548042716729e-02, 1.401921124636750e+00, 0.000000000000000e+00, -2.971948889478728e-01, -6.610210198862612e+01, 0.000000000000000e+00, -1.810525144058167e+00, -3.017328352285709e-01, 0.000000000000000e+00, -2.828833754341816e-01, -1.322579374261033e+00, 0.000000000000000e+00, -1.885868156853848e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
