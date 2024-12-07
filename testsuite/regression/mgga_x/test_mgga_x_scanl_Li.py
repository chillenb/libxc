
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_scanl_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_scanl", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.862004712840425e+00, -1.366274442292198e+00, -3.286291578446273e-01, -1.627915360702432e-01, -7.184056519569061e-02, -9.305351407635196e-03, -7.304891950196949e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_scanl_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_scanl", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.114876606603398e+00, -2.117873876170543e+00, -1.551018087143950e+00, -1.552041849539265e+00, -4.635909013777151e-01, -4.634039553401222e-01, -1.959375935658680e-01, -9.035233511302719e-03, -9.908290499170663e-02, "nan", -9.660468466843063e-03, "nan", -6.332774973155975e-04, "nan"]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_scanl_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_scanl", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-6.315299612161085e-04, 0.000000000000000e+00, -6.269959496645563e-04, -1.422365610003549e-03, 0.000000000000000e+00, -1.419920923765395e-03, 1.227274353516572e-02, 0.000000000000000e+00, 1.227505756795529e-02, -1.030958943882264e+01, 0.000000000000000e+00, 2.262818041024977e+01, 7.211252344146639e+00, 0.000000000000000e+00, "nan", 2.083079590500356e+01, 0.000000000000000e+00, "nan", 4.131756441272822e+04, 0.000000000000000e+00, "nan"]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_scanl_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_scanl", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [5.497871362558540e-03, 5.476224141509354e-03, 4.505129222656144e-03, 4.508081298788816e-03, 0.000000000000000e+00, 0.000000000000000e+00, 6.256313952888444e-02, 3.976326735678229e-05, 0.000000000000000e+00, "nan", 4.357015435682311e-05, "nan", 2.376314974109887e-05, "nan"]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_scanl_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_scanl", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
