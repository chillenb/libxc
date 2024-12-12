
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_scan_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_scan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-2.037410945304787e+00, -1.413071609489060e+00, -3.260950376753207e-01, -1.840457959428116e-01, -7.184056186855332e-02, -5.937158472811017e-03, -2.193682702841492e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_scan_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_scan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.672608816000404e+00, -2.675091757507745e+00, -1.830982200031653e+00, -1.832651562064706e+00, -2.608845187582384e-01, -3.099474785811609e-01, -2.432148045416103e-01, 1.887701663493957e+00, -8.234706980260578e-02, 7.274271884525782e+00, -9.521286809595747e-03, 1.874986943535569e+00, -4.692758855616774e-05, 7.511557686454695e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_scan_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_scan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-2.073703770459850e-04, 0.000000000000000e+00, -2.066203958803649e-04, -8.120829601150529e-04, 0.000000000000000e+00, -8.085213327197871e-04, -2.530726233237059e-01, 0.000000000000000e+00, -1.858831177530910e-01, -3.311990934301652e+00, 0.000000000000000e+00, -4.860816314658011e+04, -9.044716184138645e+01, 0.000000000000000e+00, -1.475263471197845e+10, 2.043279899229184e+01, 0.000000000000000e+00, -4.135922010229128e+04, 3.271621456622518e+04, 0.000000000000000e+00, -1.452382627711716e+11])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_scan_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_scan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([1.113644929835755e-02, 1.112620842050660e-02, 1.489543910846029e-02, 1.487085220818612e-02, 6.377943703150706e-02, 4.764894994054759e-02, 1.293931624782594e-01, 6.211671012042239e-01, 2.335477623312419e-01, 6.010756650290325e+00, 2.011490007075790e-11, 6.012941965513501e-01, 2.422576971428648e-23, 6.335760566444134e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
