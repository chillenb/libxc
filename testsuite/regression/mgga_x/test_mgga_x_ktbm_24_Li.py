
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_ktbm_24_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_24", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.998961589622010e+00, -1.317335669955434e+00, -2.158596485679645e-01, -1.839876218607651e-01, -4.917229365578080e-02, -8.539004251506218e-03, -1.589929683433533e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_ktbm_24_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_24", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.826184577498086e+00, -2.828904425844031e+00, -1.988103655661467e+00, -1.989726325365160e+00, -3.005574841547567e-01, -3.013885258260169e-01, -2.539538750010507e-01, -1.147787818448534e-02, -7.616094622258915e-02, -3.639053423911403e-04, -1.206587073793952e-02, -1.198222930136482e-02, -2.430676600381252e-04, -1.614239957779551e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_24_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_24", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-8.044917105251424e-04, 0.000000000000000e+00, -8.017901206902894e-04, -3.041580844101855e-03, 0.000000000000000e+00, -3.033352029286365e-03, -7.134121645420652e-02, 0.000000000000000e+00, -7.403493916001479e-02, -1.253811752755743e+01, 0.000000000000000e+00, 1.555502924139032e+01, -9.701297184425658e+01, 0.000000000000000e+00, 3.919473376912650e+04, 2.909694753835536e-01, 0.000000000000000e+00, 1.390286141557889e+01, 5.935293414432089e-01, 0.000000000000000e+00, -2.599675038946072e+05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_24_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_24", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([2.314253111847839e-02, 2.311005925615230e-02, 3.400938429519251e-02, 3.398725890168480e-02, 2.327426572568749e-03, 2.516705311118391e-03, 2.434997845150520e-01, -1.980269954487709e-04, 8.873569313726559e-02, -1.596919295214328e-05, -4.320923279955981e-06, -2.013134038640887e-04, -7.206399077110318e-11, -2.150040594893106e-12])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
