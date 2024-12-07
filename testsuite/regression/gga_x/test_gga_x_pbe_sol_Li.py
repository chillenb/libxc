
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_pbe_sol_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_pbe_sol", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.769511365532051e+00, -1.250974826063028e+00, -3.829031445540636e-01, -1.586782740380967e-01, -7.446002830672675e-02, -2.053487612911458e-02, -3.838585506946353e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_pbe_sol_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_pbe_sol", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.271436373651630e+00, -2.273554634410163e+00, -1.551369155524387e+00, -1.552744784570118e+00, -3.613824851305918e-01, -3.614024610744609e-01, -2.069665508742861e-01, -2.608209836802266e-02, -7.462460130118019e-02, -8.296405943619839e-04, -2.741783247940163e-02, -2.722268016937075e-02, -5.541548354978823e-04, -3.939539036232006e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_pbe_sol_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_pbe_sol", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.488911743551942e-04, 0.000000000000000e+00, -1.483686929351225e-04, -6.107084450449742e-04, 0.000000000000000e+00, -6.087008533564460e-04, -7.233303915480400e-02, 0.000000000000000e+00, -7.223334762253311e-02, -2.270477206506560e+00, 0.000000000000000e+00, -4.928470353865910e-01, -5.395187197228081e+01, 0.000000000000000e+00, -3.158643964366127e+00, -5.007266738583358e-01, 0.000000000000000e+00, -4.676383552240165e-01, -2.299383354547798e+00, 0.000000000000000e+00, -3.291333136549075e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
