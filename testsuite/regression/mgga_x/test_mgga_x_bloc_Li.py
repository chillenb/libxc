
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_bloc_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_bloc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.750859186495887e+00, -1.209543471002415e+00, -2.977463914237423e-01, -1.585801929122938e-01, -6.345454625449133e-02, -2.055078483688031e-02, -3.505641557988976e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_bloc_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_bloc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.349540698002525e+00, -2.351673745982466e+00, -1.628193158117273e+00, -1.629613442017412e+00, -3.936157016237703e-01, -3.932657660380642e-01, -2.124450880323105e-01, -2.611569208058221e-02, -8.274698585740911e-02, -8.296433204580664e-04, -2.750705699884594e-02, -2.725989618662414e-02, -5.541564195118446e-04, -2.260030367871288e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_bloc_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_bloc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-3.434646249718020e-04, 0.000000000000000e+00, -3.408116814941779e-04, -9.617640769642784e-04, 0.000000000000000e+00, -9.597565739321190e-04, -3.978292877904628e-02, 0.000000000000000e+00, -4.049398659749163e-02, -7.459170666448908e+00, 0.000000000000000e+00, -2.780506390221201e-01, -3.256406113470528e+01, 0.000000000000000e+00, -1.776498165546454e+00, -1.183371769128082e-04, 0.000000000000000e+00, -2.638791256248716e-01, -8.108568371384401e-11, 0.000000000000000e+00, -1.557397863841548e+11])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_bloc_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_bloc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([2.323677506163095e-03, 2.304817011437339e-03, 2.364145176797173e-03, 2.368349765299592e-03, -6.355538362972272e-04, -6.726087519172267e-04, 2.683296950295916e-02, 6.652326229178553e-11, -1.556610338874278e-02, 3.476993767558143e-17, 7.573558693176799e-16, 7.620791406752818e-11, 1.425570721799182e-33, -5.495081988232815e-13])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
