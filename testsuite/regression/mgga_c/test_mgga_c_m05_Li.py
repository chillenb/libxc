
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_m05_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m05", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-5.593075819810847e-02, -5.312835608757441e-02, 3.172466139429175e-02, -6.911693296844165e-04, -1.259011089808138e-08, -3.902664384046117e-03, -8.267900129372722e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_m05_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m05", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-5.706410673905634e-02, -5.695227778525173e-02, -3.200384496756575e-02, -3.175949808042276e-02, -4.270963569610033e-02, -4.366948632921999e-02, -1.539196967355446e-02, -1.144197166910400e-01, 4.826288272394301e-03, -7.929786201910576e-02, -5.152275823561595e-03, -5.097322052038485e-03, -7.819596986600185e-05, -2.147358840740951e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_m05_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m05", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [2.159688664106861e-05, 0.000000000000000e+00, 2.199034897222780e-05, -3.217078377428074e-04, 0.000000000000000e+00, -3.211279228217300e-04, 3.515936662760404e-02, 0.000000000000000e+00, 3.546571087175308e-02, 1.752471212061693e+01, 0.000000000000000e+00, -1.213704077451338e+02, -2.816278300654700e+01, 0.000000000000000e+00, 5.406867054946838e+04, -1.426692874086088e+00, 0.000000000000000e+00, 1.359364964189487e+01, -2.178749267571585e+00, 0.000000000000000e+00, 3.317991329331454e+05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_m05_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m05", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_m05_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m05", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-2.404843004984148e-03, -2.429797871357314e-03, 5.028968894844269e-03, 5.031385709364677e-03, 1.173699719643959e-03, 1.229046680521333e-03, -6.084593066888899e-01, -2.222554551342977e-04, 6.735059717564701e-02, -2.874695601709191e-05, -9.726448130357509e-08, -2.235763763059440e-04, -1.050375307490690e-15, -1.447603883807046e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
