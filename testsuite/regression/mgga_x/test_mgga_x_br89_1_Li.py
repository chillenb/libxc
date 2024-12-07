
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_br89_1_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_br89_1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.382820457993923e+00, -1.207209722365355e+00, -4.835212014151215e-01, -1.469642357620209e-01, -8.663174096446655e-02, -1.681406638089178e-01, -4.498275941926601e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_br89_1_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_br89_1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.870072500051874e+00, -1.871845306341474e+00, -1.735707668326732e+00, -1.737315269220272e+00, -4.055995485807161e-01, -4.054730836946405e-01, -2.236285660481611e-01, -7.646448305005019e-02, -8.185585753592392e-02, -6.783089044196874e-03, -7.636357531806412e-02, -7.604712050371570e-02, -3.583135351455155e-03, -1.532172015683444e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_br89_1_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_br89_1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-7.727521932655286e-07, 0.000000000000000e+00, -7.727335838077504e-07, -1.235412425827060e-03, 0.000000000000000e+00, -1.232068714912684e-03, -1.007359192910650e-01, 0.000000000000000e+00, -1.007116503193230e-01, -2.576156457437535e+00, 0.000000000000000e+00, -9.485284552156190e+02, -7.973761536773010e+01, 0.000000000000000e+00, -4.620739495427624e+07, -8.347087965890898e+02, 0.000000000000000e+00, -8.431973174426788e+02, -1.439269995871519e+08, 0.000000000000000e+00, -4.409896219309520e+08]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_br89_1_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_br89_1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [-1.002432169242189e-05, -1.005120060741989e-05, -5.363577177587311e-03, -5.362730434381820e-03, -6.063451385509231e-03, -6.054259907204436e-03, -2.472288217549737e-02, -3.028580083060354e-03, -4.767265084867655e-02, -4.706632279114499e-03, -3.099101232650510e-03, -3.062805265685836e-03, -4.368762100705401e-03, -4.809344240465720e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_br89_1_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_br89_1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [4.009728676968758e-05, 4.020480242967956e-05, 2.145430871034920e-02, 2.145092173752728e-02, 2.425380554203693e-02, 2.421703962881774e-02, 9.889152870198947e-02, 1.211432033224142e-02, 1.906906033947062e-01, 1.882652911645799e-02, 1.239640493060204e-02, 1.225122106274334e-02, 1.747504840282161e-02, 1.923737696186288e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
