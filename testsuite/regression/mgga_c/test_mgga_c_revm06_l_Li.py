
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_revm06_l_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_revm06_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-7.807226375460329e-02, -6.189492959983724e-02, 3.573336398711442e-03, 6.160877260032623e-05, 6.980240533394780e-08, 2.185042407592593e-02, 5.559074157556135e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_revm06_l_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_revm06_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-9.740960726234804e-02, -9.705503286644221e-02, -5.565647225204880e-02, -5.520903642644515e-02, -3.278936688125775e-02, -3.443131647732620e-02, -3.646276032001767e-03, 6.116079981349355e-01, 3.609056758150784e-02, 3.681605806025811e-01, 3.373827317339243e-02, 3.181095595179940e-02, 6.713503424454968e-04, 1.165144498667184e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_revm06_l_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_revm06_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-3.216588205338047e-05, 0.000000000000000e+00, -3.185740390481242e-05, -4.768282525047531e-04, 0.000000000000000e+00, -4.759466677663173e-04, -4.134791295283959e-02, 0.000000000000000e+00, -4.381859305923314e-02, 4.153595815265796e+00, 0.000000000000000e+00, -1.158707843985125e+02, -2.105987252483305e+02, 0.000000000000000e+00, -1.007218477203965e+06, -3.331652235115831e+00, 0.000000000000000e+00, -2.229745601455532e+02, -1.353343714195623e+01, 0.000000000000000e+00, -4.813640380061207e+06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_revm06_l_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_revm06_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([4.339209797575391e-03, 4.312856967751758e-03, 1.137618468933535e-02, 1.136217113597137e-02, 2.244699118454496e-02, 2.376335371617449e-02, -1.427047365699135e-01, 2.973435729122395e-03, 5.036417575247544e-01, 4.160008046322699e-04, 1.610026434421570e-06, 3.261182558426051e-03, 1.523779516207745e-14, 2.099874397412567e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
