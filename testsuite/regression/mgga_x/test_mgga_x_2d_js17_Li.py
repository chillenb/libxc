
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_2d_js17_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_2d_js17", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-3.946266036862000e+00, -2.276288494485467e+00, -1.994453295810489e-01, -1.068894207136402e-01, -1.149321453655024e-02, -1.035616154371908e+01, -2.636954856239478e+03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_2d_js17_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_2d_js17", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-5.928801289473816e+00, -5.936028262365340e+00, -3.432822951538366e+00, -3.437189400742700e+00, -4.387879784953173e-01, -4.376767098685226e-01, -1.617169990226911e-01, 6.055499265336556e-01, -4.788514492556539e-02, 9.789090916838943e-01, 6.130173342573007e+00, 6.071858569279993e-01, 1.398001690035674e+03, -9.385354945053629e+02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_2d_js17_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_2d_js17", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([1.699477036302663e-03, 0.000000000000000e+00, 1.700172111202497e-03, 8.698330361328711e-03, 0.000000000000000e+00, 8.665692177879105e-03, -9.726222142734191e-01, 0.000000000000000e+00, -8.226344762762912e-01, 8.139849288362988e+01, 0.000000000000000e+00, -2.971239436125520e+04, -8.653277867249168e+03, 0.000000000000000e+00, -3.799588658513350e+09, -5.328156091101364e+03, 0.000000000000000e+00, -2.551515153022634e+04, -1.230192632685931e+08, 0.000000000000000e+00, -5.208478118912159e+24])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_2d_js17_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_2d_js17", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([1.450796869443232e-03, 1.449713371593992e-03, 2.509313579799927e-03, 2.506113308884213e-03, 2.131122667651162e-02, 2.132482850038483e-02, 5.337191547637748e-02, 1.707160856468683e-02, 2.138322433992461e-01, 6.964216324209926e-02, 3.559717620045676e-03, 1.667773696857365e-02, 6.719332240473582e-04, 1.583242545584270e+03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
