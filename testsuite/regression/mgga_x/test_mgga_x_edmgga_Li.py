
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_edmgga_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_edmgga", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.377240932520893e+00, -1.234905913687411e+00, -4.947455676945538e-01, -1.503655346207629e-01, -8.673677864984147e-02, -1.566142620058618e-01, -4.174868999706228e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_edmgga_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_edmgga", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.854586827154979e+00, -1.856323528650963e+00, -1.752861222194393e+00, -1.754389143723399e+00, -3.957573805431987e-01, -3.957155070876247e-01, -2.345157529111785e-01, -7.401826255510437e-02, -7.709054918486691e-02, -5.958153179432902e-03, -7.421925468889071e-02, -7.388004846748555e-02, -2.965837323516025e-03, -1.097549429674290e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_edmgga_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_edmgga", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-5.435834088880098e-07, 0.000000000000000e+00, -5.436806742388414e-07, -1.042870483638507e-03, 0.000000000000000e+00, -1.039330326372845e-03, -1.111834162043802e-01, 0.000000000000000e+00, -1.111247709687835e-01, -3.166607599665189e+00, 0.000000000000000e+00, -8.642065587281720e+02, -9.136035469371379e+01, 0.000000000000000e+00, -4.302887929163027e+07, -7.601309636668367e+02, 0.000000000000000e+00, -7.679583538792695e+02, -1.344186328164115e+08, 0.000000000000000e+00, -4.124427680920017e+08])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_edmgga_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_edmgga", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = numpy.asarray([-7.051490768767839e-06, -7.071833860544934e-06, -4.527651016200751e-03, -4.523812921433237e-03, -6.692302445587491e-03, -6.680242488733192e-03, -3.038932917157857e-02, -2.759346603702367e-03, -5.462152674918462e-02, -4.382872317483164e-03, -2.822209153781553e-03, -2.789509455773085e-03, -4.080145006575931e-03, -4.498017986363379e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_edmgga_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_edmgga", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([2.820596307507136e-05, 2.828733544217974e-05, 1.811060406480300e-02, 1.809525168573295e-02, 2.676920978234996e-02, 2.672096995493277e-02, 1.215573166863143e-01, 1.103738641480947e-02, 2.184861069967385e-01, 1.753148926993266e-02, 1.128883661512621e-02, 1.115803782309234e-02, 1.632058002630372e-02, 1.799207194545352e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
