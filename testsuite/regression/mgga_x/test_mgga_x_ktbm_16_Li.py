
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_ktbm_16_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_16", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-2.134596943800216e+00, -1.493480654179448e+00, -3.112491449421522e-01, -1.913800917425063e-01, -7.067839120920545e-02, -1.164255501410962e-02, -2.179744639688859e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_ktbm_16_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_16", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.633326729295082e+00, -2.635888314090309e+00, -1.762661972014900e+00, -1.764072954279788e+00, -3.946633447163337e-01, -3.960418415919668e-01, -2.437266463024530e-01, -1.430258052306268e-02, -8.844282576918890e-02, -4.535958407693190e-04, -1.563787738141097e-02, -1.493078048833553e-02, -3.151323117969620e-04, -2.153882176626742e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_16_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_16", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-7.649389047691340e-04, 0.000000000000000e+00, -7.623632768742068e-04, -2.974553537182628e-03, 0.000000000000000e+00, -2.966735588133959e-03, -5.925298796811392e-02, 0.000000000000000e+00, -6.266590467793565e-02, -1.173586371825015e+01, 0.000000000000000e+00, -1.205681984195656e+01, -9.171100513052335e+01, 0.000000000000000e+00, -3.013144101996982e+04, 3.104394805236688e-01, 0.000000000000000e+00, -1.078168579221065e+01, 6.598620054506443e-01, 0.000000000000000e+00, -1.364091096895257e+05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_16_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_16", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([3.340006572892937e-02, 3.336911554235284e-02, 4.926413383063440e-02, 4.924902429632565e-02, 1.869070917933935e-02, 2.037934316865632e-02, 3.582407657768397e-01, 1.542908937342560e-04, 3.023976855998473e-01, 1.227667786070100e-05, -9.452762335005666e-08, 1.569814456267023e-04, -6.344032114580204e-16, 5.950606981288582e-06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
