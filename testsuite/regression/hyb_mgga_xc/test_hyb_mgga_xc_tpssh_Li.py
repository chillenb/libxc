
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_xc_tpssh_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_tpssh", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.821369507379325e+00, -1.270660287242413e+00, -3.639842619762018e-01, -1.616295306121813e-01, -6.898501322743837e-02, -1.849003913742507e-02, -3.454728280782658e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_xc_tpssh_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_tpssh", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.271893582410278e+00, -2.273816211871611e+00, -1.624034787360269e+00, -1.625581605878063e+00, -3.341652730744593e-01, -3.348806398185841e-01, -2.216298553946243e-01, -4.196384035468823e-01, -6.978316619183657e-02, -1.041496327580952e-02, -2.471151574982690e-02, -2.453395649983790e-02, -4.987399757697095e-04, -3.545587814148309e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_tpssh_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_tpssh", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-8.080243234331087e-04, 1.933562908123026e-04, -8.072200646621115e-04, -1.215728840553958e-03, 3.188370920880546e-04, -1.213324333669415e-03, -6.361141916910236e-02, -2.337108015954634e-02, -6.297315076541993e-02, 7.688087981010813e+00, 8.871800675491191e+01, 4.553404141547958e+02, -4.174098405843233e+01, 3.144581314832332e+01, 6.254055427936683e+04, -2.538982464707851e-01, -2.079140542073983e-04, -2.370455612335603e-01, -1.163898921091821e+00, 3.213895784120031e-06, -1.666001523341533e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_tpssh_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_tpssh", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_tpssh_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_tpssh", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [3.897029411906661e-02, 3.908572195973997e-02, 2.091761750545627e-02, 2.098973529225738e-02, 1.011928298343618e-03, 9.363035327908701e-04, -2.260847068640801e-01, -1.137154461635105e+00, -2.613754797781540e-02, -3.762624499430634e-02, 2.608898560619454e-14, 6.929946622422024e-11, -9.070435226639114e-32, 5.436989690300710e-19]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
