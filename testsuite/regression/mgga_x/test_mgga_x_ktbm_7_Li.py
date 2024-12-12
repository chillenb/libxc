
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_ktbm_7_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_7", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-2.046491782736548e+00, -1.431447811978452e+00, -3.256714991325638e-01, -1.838656318901190e-01, -7.202681700113778e-02, -1.205764124820034e-02, -2.207402948429075e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_ktbm_7_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_7", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.572094977154652e+00, -2.574581002047386e+00, -1.724166604819771e+00, -1.725642298920296e+00, -3.944461060126431e-01, -3.955280325388320e-01, -2.370415887840518e-01, -1.482519144106902e-02, -8.544380136981715e-02, -4.701911622878194e-04, -1.560474523924559e-02, -1.547630262055204e-02, -3.142958096812235e-04, -2.232684588069024e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_7_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_7", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-5.603037115022882e-04, 0.000000000000000e+00, -5.583834145751306e-04, -2.228189060027059e-03, 0.000000000000000e+00, -2.221874270782933e-03, -6.864133415571907e-02, 0.000000000000000e+00, -7.151614406840280e-02, -8.546838751724472e+00, 0.000000000000000e+00, -2.633170778272561e+01, -8.893768664633079e+01, 0.000000000000000e+00, -6.600441304402812e+04, -2.340622420987477e-01, 0.000000000000000e+00, -2.354249109634549e+01, -4.652294422373547e-01, 0.000000000000000e+00, -2.988127782117911e+05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_7_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_7", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([2.440851343518680e-02, 2.438734805173239e-02, 3.500374953659997e-02, 3.499130899313034e-02, 1.867513737831709e-02, 2.010484389814794e-02, 2.667469190276605e-01, 3.364930783725260e-04, 2.516214578611897e-01, 2.689257383210826e-05, 7.085033232274340e-08, 3.422679885227018e-04, 4.472775935370398e-16, 1.303516998520096e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
