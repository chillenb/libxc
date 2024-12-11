
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_pw86_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_pw86", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.788659607193134e+00, -1.293191982933969e+00, -4.224726023436478e-01, -1.591917603136986e-01, -8.242583765267496e-02, -4.981568975466691e-02, -3.866713786712611e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_pw86_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_pw86", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.196396791857767e+00, -2.198624723982786e+00, -1.456723333474594e+00, -1.458056372386369e+00, -4.107347600946448e-01, -4.107473670984644e-01, -2.040379408901762e-01, -3.887821697466316e-02, -8.121007642257914e-02, -4.046451166911972e-03, -4.003357148883172e-02, -4.011903588462037e-02, -3.278704769482822e-03, -2.577587840211097e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_pw86_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_pw86", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-3.212879111569754e-04, 0.000000000000000e+00, -3.199993094904709e-04, -1.405991713317283e-03, 0.000000000000000e+00, -1.401547801944923e-03, -7.398950702196591e-02, 0.000000000000000e+00, -7.388285880829808e-02, -4.022528504918088e+00, 0.000000000000000e+00, -2.457455869007842e+02, -6.278271487197250e+01, 0.000000000000000e+00, -2.051511656886892e+06, -2.190874295948196e+02, 0.000000000000000e+00, -2.169944613344269e+02, -4.759348657598858e+06, 0.000000000000000e+00, -1.245946420782150e+07])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
