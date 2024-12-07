
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_qtp17_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_qtp17", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-7.271511216241662e-01, -5.186785051513698e-01, -1.122487018921780e-01, -6.659653855009298e-02, -2.852103540102904e-02, -8.791802601023585e-03, -2.184379574639887e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_qtp17_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_qtp17", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-9.553848982594569e-01, -9.560019019927097e-01, -6.850673960743782e-01, -6.854177931666001e-01, -2.453198536515200e-01, -2.456069249358110e-01, -8.730777696796957e-02, -9.637156689348658e-02, -3.708595460563237e-02, -4.004485536102022e-02, -1.133267488230034e-02, -1.137865440282262e-02, -2.869368253276484e-04, -2.904046641804044e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_qtp17_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_qtp17", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [5.281135584412428e-06, 4.178252337481369e-06, 5.181877237031280e-06, 3.242664345705641e-05, 2.917553431398870e-05, 3.190321763918229e-05, 2.768861105358829e-02, 3.819010290868950e-02, 2.781532675616890e-02, -3.355764937893764e-04, 3.676907815562432e+00, 2.758838039210102e+00, 1.742740340209228e-06, 1.885551787463729e+01, 1.414163996275170e+01, 3.269546824267468e-02, 6.348877857422126e-02, 3.286341928183299e-02, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
