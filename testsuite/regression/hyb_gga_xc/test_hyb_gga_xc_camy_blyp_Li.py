
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_camy_blyp_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_camy_blyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.332986910765894e+00, -9.173660541222559e-01, -1.494912647258070e-01, -4.388347433446025e-02, -6.012014754413306e-03, -2.125813496726139e-03, -3.003803580999289e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_camy_blyp_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_camy_blyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.709120444305975e+00, -1.710621563955989e+00, -1.148193802646884e+00, -1.149087142255587e+00, -2.978959878678515e-01, -2.982655210449881e-01, -7.029765609792414e-02, -8.231162313486552e-02, -1.065307679181685e-02, -3.029946692596101e-02, -2.783600692096201e-03, -2.872601944112721e-03, -2.079765025506268e-05, -9.360867686672843e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_camy_blyp_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_camy_blyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.763268844283253e-04, 5.222815421851711e-06, -1.758752974838314e-04, -5.951992485611571e-04, 3.646941789248587e-05, -5.940012674900794e-04, 5.368468816822305e-03, 4.773762863586187e-02, 5.560246701384837e-03, -6.647136351843136e-01, 4.596134769453040e+00, 3.448099513568591e+00, -1.177509901914730e+00, 2.356939734329661e+01, 1.767704985390812e+01, 4.035022188209863e-02, 7.936097321777658e-02, 4.059344615877507e-02, -2.476981568285759e-08, 0.000000000000000e+00, -1.198271898660026e-08])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
