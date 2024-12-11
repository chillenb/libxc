
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_m06_2x_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m06_2x", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-6.916551642429998e-02, -7.035261411589139e-02, -1.182938674369627e-01, 1.468918225053278e-02, -5.469541796564105e-02, 1.541025490975264e-02, 2.697223242270711e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_m06_2x_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m06_2x", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-6.091143706157705e-02, -6.011588558009946e-02, -5.129941465694382e-02, -5.081867747994210e-02, -5.663485407814688e-02, -4.961626602230133e-02, 2.832911778252075e-02, 5.297023659328500e-01, 1.472902318192816e-02, 3.356106682941292e-01, 3.282388119400161e-03, 4.163137738511171e-03, -3.431978144901863e-04, 8.030014582193025e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_m06_2x_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m06_2x", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([2.681214331971679e-04, 0.000000000000000e+00, 2.705304093535886e-04, 2.259509365863608e-03, 0.000000000000000e+00, 2.247127324251852e-03, 2.517795727553533e+00, 0.000000000000000e+00, 2.499545934758825e+00, 2.038928993107143e+00, 0.000000000000000e+00, 5.772863628191158e+02, 3.635457011522532e+03, 0.000000000000000e+00, 1.631757259859822e+06, 7.336297262582846e+00, 0.000000000000000e+00, 3.475473493277607e+02, 2.493128764325716e+01, 0.000000000000000e+00, 1.669261289008757e+14])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_m06_2x_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m06_2x", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-2.156117902366983e-03, -2.188823793483045e-03, -3.870831885295053e-03, -3.889939756114966e-03, -1.463908352587895e-02, -1.571052857043763e-02, -3.245615926809332e-01, -5.605394307605965e-03, -6.709517559049391e-01, -6.581649496988900e-04, -1.089102725759498e-04, -5.049639369636720e-03, -3.027058575100504e-09, 7.981427041210718e-09])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
